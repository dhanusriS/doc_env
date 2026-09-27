"""Educational intercept-and-resend attack simulation for PCS-CV-QKD.

This is a simplified discrete-modulation demonstration, not a composable
security proof. Eve intercepts a selected fraction of Bob's channel outputs,
nearest-point decodes the attenuated quadratures, prepares that constellation
point again, and Bob receives an independent resend measurement noise sample.

The attack is intentionally applied to samples, not by adding an arbitrary
constant to the key-rate equation. The resulting Bob samples are used for
observed parameter estimation and the existing finite-size SKR calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.constellation import PCSConstellation
from core.parameter_estimation import ParameterEstimator


@dataclass
class EveAttackResult:
    """Samples and observed statistics from one attack experiment."""

    attack_probability: float
    seed: int
    attacked_symbols: int
    total_symbols: int
    x_alice: np.ndarray
    p_alice: np.ndarray
    x_bob: np.ndarray
    p_bob: np.ndarray
    bob_indices: np.ndarray
    attack_noise: float
    observed_qber: float
    estimated_T: float
    estimated_xi: float
    T_min: float
    xi_max: float
    noise_budget: dict[str, float]


def _nearest_indices(
    x_values: np.ndarray,
    p_values: np.ndarray,
    constellation: PCSConstellation,
) -> np.ndarray:
    """Decode received quadratures to the nearest constellation point."""
    distances = (
        (x_values[:, None] - constellation.points_x[None, :]) ** 2
        + (p_values[:, None] - constellation.points_p[None, :]) ** 2
    )
    return np.argmin(distances, axis=1)


def estimate_observed_channel(
    x_alice: np.ndarray,
    p_alice: np.ndarray,
    x_bob: np.ndarray,
    p_bob: np.ndarray,
    modulation_variance: float,
    nu_el: float,
    eps_pe: float = 1e-10,
) -> tuple[float, float, float, float]:
    """Estimate channel parameters for known discrete-modulation samples.

    The existing estimator is retained for the protocol runner. This estimator
    uses the known modulation variance directly, which is appropriate for the
    discrete PCS samples used by this attack demonstration:

        T_hat = Cov(A,B)^2 / Var(A)^2
        xi_hat = (Var(B - sqrt(T_hat) A) - 1 - nu_el) / T_hat

    The returned bounds are conservative demo bounds based on Gaussian standard
    errors; they are not a replacement for a composable discrete-modulation
    parameter-estimation proof.
    """
    x_a = np.concatenate((x_alice, p_alice))
    x_b = np.concatenate((x_bob, p_bob))
    # modulation_variance is X^2 + P^2; each quadrature carries half of it.
    variance = max(float(modulation_variance) / 2.0, 1e-12)
    covariance = float(np.mean((x_a - x_a.mean()) * (x_b - x_b.mean())))
    t_hat = float(np.clip((covariance / variance) ** 2, 1e-4, 1.0))
    residual = x_b - np.sqrt(t_hat) * x_a
    residual_variance = float(np.var(residual, ddof=1))
    xi_hat = max((residual_variance - 1.0 - nu_el) / t_hat, 0.0)

    n = max(x_a.size, 1)
    z = 6.5 if eps_pe <= 1e-10 else 4.0
    se_t = z * np.sqrt(max(t_hat * (1.0 - t_hat), 1e-12) / n)
    se_xi = z * np.sqrt(max(2.0 * xi_hat**2 + 4.0, 1e-12) / n)
    return t_hat, xi_hat, max(t_hat - se_t, 1e-4), xi_hat + se_xi


def simulate_intercept_resend(
    x_alice: np.ndarray,
    p_alice: np.ndarray,
    x_bob_baseline: np.ndarray,
    p_bob_baseline: np.ndarray,
    constellation: PCSConstellation,
    transmittance: float,
    baseline_noise_variance: float,
    attack_probability: float,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply a reproducible simplified intercept-and-resend attack.

    Eve receives the attenuated signal represented by Bob's baseline samples.
    On attacked symbols she nearest-point decodes an estimate of the original
    constellation point, prepares that point, and sends it toward Bob. Bob then
    receives independent baseline measurement noise for the resend. Unattacked
    samples are returned byte-for-byte unchanged from the baseline channel.
    """
    probability = float(np.clip(attack_probability, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    attacked = rng.random(x_alice.size) < probability
    x_bob = np.array(x_bob_baseline, copy=True)
    p_bob = np.array(p_bob_baseline, copy=True)

    if not np.any(attacked):
        return x_bob, p_bob, attacked, 0.0

    sqrt_t = np.sqrt(max(transmittance, 1e-12))
    eve_x = x_bob_baseline[attacked] / sqrt_t
    eve_p = p_bob_baseline[attacked] / sqrt_t
    eve_indices = _nearest_indices(eve_x, eve_p, constellation)
    resend_sigma = np.sqrt(max(baseline_noise_variance, 1e-12))
    x_bob[attacked] = (
        sqrt_t * constellation.points_x[eve_indices]
        + rng.normal(0.0, resend_sigma, size=eve_indices.size)
    )
    p_bob[attacked] = (
        sqrt_t * constellation.points_p[eve_indices]
        + rng.normal(0.0, resend_sigma, size=eve_indices.size)
    )
    return x_bob, p_bob, attacked, float(np.var(x_bob[attacked] - x_bob_baseline[attacked]))


def run_eve_attack_experiment(
    constellation: PCSConstellation,
    channel,
    distance_km: float,
    nu: float,
    attack_probability: float,
    symbols: int = 20_000,
    seed: int = 2026,
    xi_override: float | None = None,
    eps_pe: float = 1e-10,
) -> EveAttackResult:
    """Generate Alice/Bob samples, apply Eve, and estimate observed statistics."""
    if symbols < 100:
        raise ValueError("symbols must be at least 100 for attack estimation")

    np.random.seed(seed)
    x_alice, p_alice, alice_indices = constellation.draw_symbols(symbols, nu)
    x_bob_base, p_bob_base, noise = channel.transmit(
        x_alice, p_alice, distance_km, xi_override
    )
    x_bob, p_bob, attacked, attack_noise = simulate_intercept_resend(
        x_alice,
        p_alice,
        x_bob_base,
        p_bob_base,
        constellation,
        noise["T"],
        noise["total"],
        attack_probability,
        seed=seed,
    )

    bob_indices = _nearest_indices(x_bob, p_bob, constellation)
    observed_qber = float(np.mean(bob_indices != alice_indices))
    t_hat, xi_hat, t_min, xi_max = estimate_observed_channel(
        x_alice,
        p_alice,
        x_bob,
        p_bob,
        constellation.modulation_variance(nu),
        channel.cfg.nu_el,
        eps_pe,
    )
    attack_noise = float(attack_noise)
    noise_budget = dict(noise)
    noise_budget["eve_intercept_resend"] = attack_noise
    noise_budget["observed_total"] = noise["total"] + attack_noise * float(np.mean(attacked))
    return EveAttackResult(
        attack_probability=float(np.clip(attack_probability, 0.0, 1.0)),
        seed=seed,
        attacked_symbols=int(np.count_nonzero(attacked)),
        total_symbols=symbols,
        x_alice=x_alice,
        p_alice=p_alice,
        x_bob=x_bob,
        p_bob=p_bob,
        bob_indices=bob_indices,
        attack_noise=attack_noise,
        observed_qber=observed_qber,
        estimated_T=t_hat,
        estimated_xi=xi_hat,
        T_min=t_min,
        xi_max=xi_max,
        noise_budget=noise_budget,
    )
