"""
simulation/protocol_runner.py
==============================
Orchestrates one complete PCS-CV-QKD protocol round.

NO BUGS in this module — updated only to use v2 imports and
the corrected nu_min/nu_max defaults for the normalised constellation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from core.constellation        import PCSConstellation
from core.channel              import OpticalChannel, ChannelConfig
from core.key_rate             import compute_skr, optimal_nu, SKRResult
from core.parameter_estimation import ParameterEstimator
from ai.lstm_agent             import NuAgent


@dataclass
class RoundResult:
    """All outputs from one QKD round."""
    distance_km:     float
    nu_used:         float
    N_block:         int
    T_true:          float
    xi_true:         float
    T_hat:           float
    xi_hat:          float
    T_min:           float
    xi_max:          float
    n_PE:            int
    skr_finite:      float
    skr_asymptotic:  float
    I_AB:            float
    chi_BE:          float
    VA:              float
    SNR:             float
    beta:            float
    key_bits:        int
    noise_budget:    dict = field(default_factory=dict)
    nu_next:         float = 5.0


@dataclass
class ProtocolConfig:
    """Protocol hyperparameters."""
    N_block:   int   = 500_000
    delta_PE:  float = 0.10
    eps_PE:    float = 1e-10
    eps_EC:    float = 1e-10
    eps_PA:    float = 1e-10
    M:         int   = 256


class ProtocolRunner:
    """Full PCS-CV-QKD protocol runner."""

    def __init__(
        self,
        proto_cfg:  ProtocolConfig | None = None,
        ch_cfg:     ChannelConfig  | None = None,
        agent:      NuAgent        | None = None,
    ):
        self.cfg           = proto_cfg or ProtocolConfig()
        self.constellation = PCSConstellation(self.cfg.M)
        self.channel       = OpticalChannel(ch_cfg or ChannelConfig())
        self.estimator     = ParameterEstimator(
            eps_PE=self.cfg.eps_PE,
            nu_el=self.channel.cfg.nu_el,
        )
        self.agent  = agent or NuAgent()
        self._nu_current  = 5.0   # Initialise at reasonable value for normalised constellation
        self._round_idx   = 0
        self.history: list[RoundResult] = []

    def single_round(
        self,
        distance_km:    float,
        xi_override:    float | None = None,
        use_optimal_nu: bool  = False,
    ) -> RoundResult:
        """Execute one complete QKD protocol round."""
        cfg = self.cfg
        N   = cfg.N_block
        nu  = self._nu_current

        # Step 1: Symbol generation with PCS
        x_A, p_A, _ = self.constellation.draw_symbols(N, nu)

        # Step 2: Quantum channel transmission
        x_B, p_B, noise = self.channel.transmit(x_A, p_A, distance_km, xi_override)
        T_true  = noise["T"]
        xi_true = noise["xi"]

        # Step 3: Parameter estimation
        n_PE   = max(int(cfg.delta_PE * N), 100)
        pe_idx = np.random.choice(N, size=n_PE, replace=False)
        pe     = self.estimator.estimate(
            x_A[pe_idx], x_B[pe_idx],
            p_A[pe_idx], p_B[pe_idx],
        )

        # Step 4: Key rate at worst-case parameters
        if use_optimal_nu:
            nu, skr_res = optimal_nu(
                pe.T_min, pe.xi_max,
                self.channel.cfg.nu_el,
                self.constellation, N, cfg.delta_PE,
            )
        else:
            skr_res = compute_skr(
                pe.T_min, pe.xi_max,
                nu, self.channel.cfg.nu_el,
                self.constellation, N, cfg.delta_PE,
                cfg.eps_PE, cfg.eps_EC, cfg.eps_PA,
            )

        key_bits = max(int((N - n_PE) * skr_res.skr_finite), 0)

        # Step 5: AI agent update → nu* for next block
        nu_next = self.agent.update(
            T=pe.T_hat, xi=pe.xi_hat,
            SNR=pe.SNR_hat,
            skr=skr_res.skr_finite,
        )
        self._nu_current = nu_next
        self._round_idx += 1

        result = RoundResult(
            distance_km=distance_km, nu_used=nu, N_block=N,
            T_true=T_true, xi_true=xi_true,
            T_hat=pe.T_hat, xi_hat=pe.xi_hat,
            T_min=pe.T_min, xi_max=pe.xi_max, n_PE=n_PE,
            skr_finite=skr_res.skr_finite,
            skr_asymptotic=skr_res.skr_asymptotic,
            I_AB=skr_res.I_AB, chi_BE=skr_res.chi_BE,
            VA=skr_res.VA, SNR=skr_res.SNR, beta=skr_res.beta,
            key_bits=key_bits, noise_budget=noise, nu_next=nu_next,
        )
        self.history.append(result)
        return result

    def reset(self):
        """Reset runner for a new session."""
        self._nu_current = 5.0
        self._round_idx  = 0
        self.history.clear()
        self.agent.reset()
