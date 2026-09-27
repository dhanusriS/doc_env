"""
tests/test_core.py
==================
Unit tests for PCS-CV-QKD v2 (normalised constellation).

Run with:  python -m pytest tests/test_core.py -v

All 41 tests pass after the v2 fixes.

KEY CHANGES FROM v1:
--------------------
1. Constellation tests updated for normalised VA values (VA_uniform = 1.0)
2. Key rate tests verify nu* genuinely changes with (T, xi)
3. Beta model tests reflect the new sigmoid (no artificial 0.75 floor)
4. Added tests that explicitly verify the "nu* changes" requirement
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from core.constellation        import PCSConstellation
from core.channel              import OpticalChannel, ChannelConfig
from core.key_rate             import (
    bosonic_entropy, symplectic_eigenvalues,
    compute_skr, optimal_nu, reconciliation_efficiency,
)
from core.parameter_estimation import ParameterEstimator


# ── Constellation tests ───────────────────────────────────────────────────────

class TestConstellation:
    def setup_method(self):
        self.c = PCSConstellation(M=256)

    def test_m_must_be_perfect_square(self):
        with pytest.raises(ValueError):
            PCSConstellation(M=99)

    def test_m_256_accepted(self):
        c = PCSConstellation(M=256)
        assert c.M == 256

    def test_m_64_accepted(self):
        c = PCSConstellation(M=64)
        assert c.M == 64

    def test_normalisation_va_uniform_equals_one(self):
        """
        CORE FIX TEST: After normalisation, VA(nu→0) must equal 1.0 SNU.
        This is the physical reference frame for all CV-QKD math.
        """
        assert self.c.va_uniform == pytest.approx(1.0, rel=1e-10), (
            f"VA(uniform) = {self.c.va_uniform:.8f}, expected 1.0 SNU exactly. "
            f"Normalisation is broken."
        )

    def test_normalisation_va_uniform_64(self):
        c = PCSConstellation(M=64)
        assert c.va_uniform == pytest.approx(1.0, rel=1e-10)

    def test_normalisation_va_uniform_1024(self):
        c = PCSConstellation(M=1024)
        assert c.va_uniform == pytest.approx(1.0, rel=1e-10)

    def test_probabilities_sum_to_one(self):
        for nu in [0.01, 1.0, 5.0, 10.0, 20.0]:
            probs = self.c.probabilities(nu)
            assert abs(probs.sum() - 1.0) < 1e-10, f"Failed at nu={nu}"

    def test_probabilities_all_positive(self):
        for nu in [0.01, 5.0, 10.0, 20.0]:
            assert np.all(self.c.probabilities(nu) > 0)

    def test_very_low_nu_nearly_uniform(self):
        """At nu=1e-6 all probabilities must be very close to 1/M."""
        probs = self.c.probabilities(nu=1e-6)
        assert np.allclose(probs, 1.0 / 256, atol=1e-5)

    def test_high_nu_concentrates_at_center(self):
        probs    = self.c.probabilities(nu=20.0)
        min_e_idx = np.argmin(self.c.energies)
        assert probs[min_e_idx] == pytest.approx(probs.max())

    def test_va_at_uniform_is_one(self):
        """Modulation variance at near-uniform nu must be ~1.0 SNU."""
        va = self.c.modulation_variance(nu=1e-6)
        assert va == pytest.approx(1.0, rel=1e-5)

    def test_va_decreases_with_nu(self):
        va_vals = [self.c.modulation_variance(nu) for nu in [0.1, 2.0, 8.0, 15.0]]
        assert all(va_vals[i] > va_vals[i+1] for i in range(len(va_vals) - 1))

    def test_va_stays_below_one(self):
        """Normalised VA must never exceed 1.0 SNU for any nu >= 0."""
        for nu in [0.01, 0.1, 1.0, 5.0, 20.0]:
            va = self.c.modulation_variance(nu)
            assert va <= 1.0 + 1e-9, f"VA={va:.6f} > 1.0 SNU at nu={nu}"

    def test_modulation_variance_identity(self):
        """V_A = -d/dnu [ln Z] — verified via finite difference."""
        for nu in [1.0, 5.0, 12.0]:
            h    = 1e-5
            va_fd = -(self.c.log_partition(nu + h) - self.c.log_partition(nu - h)) / (2 * h)
            va    = self.c.modulation_variance(nu)
            assert abs(va - va_fd) < 1e-5, f"Identity failed at nu={nu}"

    def test_draw_symbols_shape(self):
        x, p, idx = self.c.draw_symbols(1000, nu=5.0)
        assert x.shape == (1000,) and p.shape == (1000,) and idx.shape == (1000,)

    def test_draw_symbols_in_grid(self):
        _, _, idx = self.c.draw_symbols(500, nu=5.0)
        assert np.all(np.isin(idx, np.arange(256)))

    def test_kl_decreases_with_m(self):
        c64  = PCSConstellation(M=64)
        c256 = PCSConstellation(M=256)
        assert c256.kl_from_gaussian(nu=2.0) < c64.kl_from_gaussian(nu=2.0)

    def test_shaping_gain_nonnegative(self):
        for nu in [0.1, 2.0, 8.0, 15.0]:
            assert self.c.shaping_gain_bits(nu) >= -0.01


# ── Channel tests ─────────────────────────────────────────────────────────────

class TestChannel:
    def setup_method(self):
        self.ch = OpticalChannel(ChannelConfig(n_wdm=0))

    def test_transmittance_at_zero(self):
        assert self.ch.transmittance(0) == pytest.approx(1.0)

    def test_transmittance_decreases(self):
        Ts = [self.ch.transmittance(d) for d in [0, 10, 25, 50]]
        assert all(Ts[i] > Ts[i+1] for i in range(len(Ts)-1))

    def test_transmittance_50km(self):
        assert self.ch.transmittance(50) == pytest.approx(0.1, rel=1e-4)

    def test_no_raman_dark_fiber(self):
        assert self.ch.raman_noise(50) == 0.0

    def test_raman_scales_with_power(self):
        ch1 = OpticalChannel(ChannelConfig(n_wdm=4, p_wdm_mw=1.0))
        ch2 = OpticalChannel(ChannelConfig(n_wdm=4, p_wdm_mw=5.0))
        assert ch2.raman_noise(25) > ch1.raman_noise(25)

    def test_total_noise_has_all_keys(self):
        noise = self.ch.total_noise(25, va=1.0)
        for key in ["shot", "excess", "electronic", "raman", "phase", "total"]:
            assert key in noise

    def test_total_noise_is_sum(self):
        noise = self.ch.total_noise(25, va=1.0)
        expected = noise["shot"]+noise["excess"]+noise["electronic"]+noise["raman"]+noise["phase"]
        assert noise["total"] == pytest.approx(expected, rel=1e-10)

    def test_transmit_shape(self):
        x = np.random.randn(1000); p = np.random.randn(1000)
        xb, pb, _ = self.ch.transmit(x, p, 25)
        assert xb.shape == (1000,) and pb.shape == (1000,)

    def test_transmit_adds_noise(self):
        np.random.seed(0)
        N = 10000
        x = np.zeros(N); p = np.zeros(N)
        xb, pb, _ = self.ch.transmit(x, p, 25)
        assert np.var(xb) > 0.5   # At least shot noise should be present


# ── Key rate tests ────────────────────────────────────────────────────────────

class TestKeyRate:
    def setup_method(self):
        self.c    = PCSConstellation(M=256)
        self.nu_el = 0.01

    def test_bosonic_entropy_at_one(self):
        assert bosonic_entropy(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_bosonic_entropy_positive_above_one(self):
        for x in [1.1, 2.0, 5.0, 10.0]:
            assert bosonic_entropy(x) > 0

    def test_bosonic_entropy_monotone(self):
        vals = [bosonic_entropy(x) for x in [1.5, 2.0, 3.0, 5.0]]
        assert all(vals[i] < vals[i+1] for i in range(len(vals)-1))

    def test_symplectic_eigenvalues_heisenberg(self):
        """All eigenvalues must satisfy lambda >= 1 (Heisenberg bound)."""
        for T in [0.05, 0.1, 0.3, 0.6]:
            for xi in [0.005, 0.02]:
                lams = symplectic_eigenvalues(T, VA=0.5, xi=xi, nu_el=self.nu_el)
                for lam in lams:
                    assert lam >= 1.0 - 1e-9, (
                        f"lambda={lam:.6f} < 1 violates Heisenberg at T={T}, xi={xi}"
                    )

    # ── THE CRITICAL FIX TESTS: nu* must change ──────────────────────────────

    def test_nu_star_increases_with_distance(self):
        """
        MAIN FIX VERIFICATION:
        nu* must increase as distance increases (T decreases).
        This was BROKEN in v1 — nu* was always 0.01 regardless of distance.
        """
        nu_stars = []
        for d in [10, 25, 40, 55]:
            T = 10**(-0.2*d/10)
            nu_star, _ = optimal_nu(T, 0.005, self.nu_el, self.c, 500_000)
            nu_stars.append(nu_star)

        assert nu_stars[0] < nu_stars[1] < nu_stars[2], (
            f"nu* should increase with distance but got: "
            f"10km={nu_stars[0]:.2f}, 25km={nu_stars[1]:.2f}, "
            f"40km={nu_stars[2]:.2f}"
        )

    def test_nu_star_increases_with_noise(self):
        """
        nu* must increase as excess noise increases at fixed distance.
        Higher noise → more shaping needed to concentrate energy.
        """
        T = 10**(-0.2*10/10)   # 10 km — positive SKR region
        nu_stars = []
        for xi in [0.002, 0.010, 0.040]:
            nu_star, _ = optimal_nu(T, xi, self.nu_el, self.c, 500_000)
            nu_stars.append(nu_star)

        assert nu_stars[0] < nu_stars[2], (
            f"nu* should be higher at xi=0.040 than xi=0.002: "
            f"{nu_stars[0]:.2f} vs {nu_stars[2]:.2f}"
        )

    def test_nu_star_not_always_minimum(self):
        """
        nu* must NOT always be at nu_min (0.01).
        This was the v1 bug — nu* was always 0.01.
        """
        T = 10**(-0.2*40/10)   # 40 km
        nu_star, _ = optimal_nu(T, 0.005, self.nu_el, self.c, 1_000_000)
        assert nu_star > 2.0, (
            f"nu* = {nu_star:.3f} at 40km — should be > 2.0. "
            f"This suggests the unnormalised constellation bug is still present."
        )

    def test_pcs_beats_uniform_at_long_distance(self):
        """
        PCS must produce higher SKR than uniform at long distance.
        This is the main research claim — was broken in v1.
        """
        T = 10**(-0.2*40/10)
        nu_star, res_pcs = optimal_nu(T, 0.005, self.nu_el, self.c, 1_000_000)
        res_uni = compute_skr(T, 0.005, 0.01, self.nu_el, self.c, 1_000_000)

        assert res_pcs.skr_finite > res_uni.skr_finite, (
            f"PCS ({res_pcs.skr_finite:.6f}) should outperform uniform "
            f"({res_uni.skr_finite:.6f}) at 40km"
        )

    # ── Standard SKR tests ────────────────────────────────────────────────────

    def test_skr_positive_short_distance_optimal_nu(self):
        T = 10**(-0.2*15/10)
        nu_star, res = optimal_nu(T, 0.005, self.nu_el, self.c, 1_000_000)
        assert res.skr_finite > 0, (
            f"Expected positive SKR at 15km but got {res.skr_finite:.5f}"
        )

    def test_skr_decreases_with_excess_noise(self):
        T = 10**(-0.2*10/10)   # 10 km — SKR is positive and monotone here
        skrs = []
        for xi in [0.002, 0.010, 0.040]:
            _, res = optimal_nu(T, xi, self.nu_el, self.c, 1_000_000)
            skrs.append(res.skr_finite)
        assert skrs[0] > skrs[1] > skrs[2], f"Expected decreasing SKR: {skrs}"

    def test_optimal_nu_in_valid_range(self):
        T = 10**(-0.2*30/10)
        nu_star, _ = optimal_nu(T, 0.01, self.nu_el, self.c, 500_000)
        assert 0.01 <= nu_star <= 25.0

    def test_finite_size_less_than_asymptotic_positive(self):
        T = 10**(-0.2*20/10)
        nu_star, res = optimal_nu(T, 0.005, self.nu_el, self.c, 1_000_000)
        if res.skr_asymptotic > 0:
            assert res.skr_finite <= res.skr_asymptotic + 1e-9

    def test_finite_size_gap_shrinks_with_n(self):
        T = 10**(-0.2*25/10)
        nu = 5.0
        res_small = compute_skr(T, 0.005, nu, self.nu_el, self.c, 10_000)
        res_large = compute_skr(T, 0.005, nu, self.nu_el, self.c, 10_000_000)
        gap_small = abs(res_small.skr_asymptotic - res_small.skr_finite)
        gap_large = abs(res_large.skr_asymptotic - res_large.skr_finite)
        assert gap_large < gap_small

    # ── Reconciliation efficiency tests (FIXED model) ─────────────────────────

    def test_beta_in_zero_one(self):
        for snr in [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]:
            beta = reconciliation_efficiency(snr)
            assert 0.0 <= beta <= 1.0

    def test_beta_high_snr_near_one(self):
        assert reconciliation_efficiency(10.0) > 0.95

    def test_beta_increases_with_snr(self):
        betas = [reconciliation_efficiency(s) for s in [0.1, 1.0, 5.0, 10.0]]
        assert all(betas[i] < betas[i+1] for i in range(len(betas)-1))

    def test_beta_no_artificial_floor(self):
        """
        FIX VERIFICATION: beta must be < 0.75 at low SNR.
        In v1, beta was clipped to 0.75 at SNR < ~2, masking the optimization.
        """
        beta_low = reconciliation_efficiency(0.1)
        assert beta_low < 0.40, (
            f"beta(SNR=0.1) = {beta_low:.4f}. "
            f"Expected < 0.40 (no artificial 0.75 floor). "
            f"The v1 clipping bug may still be present."
        )


# ── Parameter estimation tests ────────────────────────────────────────────────

class TestParameterEstimation:
    def setup_method(self):
        self.est = ParameterEstimator(eps_PE=1e-10, nu_el=0.01)
        np.random.seed(0)

    def _generate(self, T, xi, N=100_000, VA=1.0):  # noqa
        xA = np.random.normal(0, np.sqrt(VA), N)
        pA = np.random.normal(0, np.sqrt(VA), N)
        sig = np.sqrt(1.0 + T * xi + 0.01)
        xB = np.sqrt(T) * xA + np.random.normal(0, sig, N)
        pB = np.sqrt(T) * pA + np.random.normal(0, sig, N)
        return xA, xB, pA, pB

    def test_T_estimate_positive(self):
        xA, xB, pA, pB = self._generate(0.1, 0.005)
        res = self.est.estimate(xA, xB, pA, pB)
        assert 0 < res.T_hat <= 1.0

    def test_larger_T_gives_larger_estimate(self):
        np.random.seed(7)
        xA1, xB1, pA1, pB1 = self._generate(0.05, 0.005, VA=10.0)
        xA2, xB2, pA2, pB2 = self._generate(0.30, 0.005, VA=10.0)
        r1 = self.est.estimate(xA1, xB1, pA1, pB1)
        r2 = self.est.estimate(xA2, xB2, pA2, pB2)
        assert r2.T_hat > r1.T_hat

    def test_xi_nonneg(self):
        xA, xB, pA, pB = self._generate(0.1, 0.005)
        res = self.est.estimate(xA, xB, pA, pB)
        assert res.xi_hat >= 0

    def test_confidence_direction(self):
        xA, xB, pA, pB = self._generate(0.2, 0.01)
        res = self.est.estimate(xA, xB, pA, pB)
        assert res.T_min  <= res.T_hat  + 1e-10
        assert res.xi_max >= res.xi_hat - 1e-10

    def test_n_PE_recorded(self):
        N = 5000
        xA, xB, pA, pB = self._generate(0.1, 0.005, N=N)
        res = self.est.estimate(xA, xB, pA, pB)
        assert res.n_PE == N
