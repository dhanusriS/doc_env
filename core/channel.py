"""
core/channel.py
===============
Full CV-QKD optical fiber channel simulator with 5-term noise budget:

    V_noise_total = shot + excess + electronic + raman + phase

All noise terms are in Shot Noise Units (SNU).

NO BUGS in this module — included unchanged from v1 for completeness.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ChannelConfig:
    """Physical channel configuration."""
    alpha_db_per_km:    float = 0.2      # SMF-28 fiber attenuation (dB/km)
    nu_el:              float = 0.01     # Detector electronic noise (SNU)
    excess_noise:       float = 0.005    # Baseline excess noise xi (SNU)
    n_wdm:              int   = 0        # Number of WDM co-channels
    p_wdm_mw:           float = 1.0      # Power per WDM channel (mW)
    raman_coeff:        float = 1e-14    # Raman coupling k_R (W^-1 SNU^-1 km^-1)
    laser_linewidth_hz: float = 1e5      # Laser linewidth (Hz)
    symbol_rate_baud:   float = 1e9      # Symbol rate (Baud)


class OpticalChannel:
    """
    Gaussian CV-QKD optical channel.

    Noise budget (all in SNU):
    --------------------------
    1. Shot noise:     1.0 SNU always — irreducible quantum floor
    2. Excess noise:   T * xi — eavesdropping + imperfections
    3. Electronic:     nu_el — detector dark current + thermal
    4. Raman:          N_c * P_c * k_R * L_eff — WDM Raman scattering
    5. Phase:          (2pi * Δν * T_s)^2 * VA — laser linewidth mismatch
    """

    def __init__(self, cfg: ChannelConfig | None = None):
        self.cfg = cfg or ChannelConfig()

    def transmittance(self, distance_km: float) -> float:
        """T = 10^(-alpha * L / 10). At 50 km: T = 0.1 exactly."""
        return float(10 ** (-self.cfg.alpha_db_per_km * distance_km / 10.0))

    def raman_noise(self, distance_km: float) -> float:
        """
        Raman noise from N_c WDM co-channels (SNU).
        Uses effective interaction length to account for fiber attenuation.
        Saturates at L_eff = 1/alpha_lin ≈ 21.7 km for standard fiber.
        """
        if self.cfg.n_wdm == 0:
            return 0.0
        alpha_lin = self.cfg.alpha_db_per_km / (10.0 * np.log10(np.e))
        l_eff     = (1 - np.exp(-alpha_lin * distance_km)) / alpha_lin
        p_watts   = self.cfg.p_wdm_mw * 1e-3
        return float(self.cfg.n_wdm * p_watts * self.cfg.raman_coeff * l_eff)

    def phase_noise(self, va: float) -> float:
        """
        Phase noise from laser linewidth mismatch (SNU).
        nu_phase = (2*pi*Δν*T_s)^2 * VA
        At standard 1 GBaud with 100 kHz laser: ~10^-12 SNU (negligible).
        """
        t_sym = 1.0 / self.cfg.symbol_rate_baud
        return float((2 * np.pi * self.cfg.laser_linewidth_hz * t_sym) ** 2 * va)

    def total_noise(
        self,
        distance_km:   float,
        va:            float,
        xi_override:   float | None = None,
    ) -> dict[str, float]:
        """
        Compute all five noise terms and their sum.

        Returns dict with keys: shot, excess, electronic, raman, phase, total, T, xi
        """
        xi   = xi_override if xi_override is not None else self.cfg.excess_noise
        T    = self.transmittance(distance_km)
        shot = 1.0
        exc  = T * xi
        elec = self.cfg.nu_el
        ram  = self.raman_noise(distance_km)
        pha  = self.phase_noise(va)
        return {
            "shot":       shot,
            "excess":     exc,
            "electronic": elec,
            "raman":      ram,
            "phase":      pha,
            "total":      shot + exc + elec + ram + pha,
            "T":          T,
            "xi":         xi,
        }

    def transmit(
        self,
        x_alice:        np.ndarray,
        p_alice:        np.ndarray,
        distance_km:    float,
        xi_override:    float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Simulate quantum channel transmission.

        Bob's received quadratures:
            x_Bob = sqrt(T) * x_Alice + N(0, sqrt(V_noise))
            p_Bob = sqrt(T) * p_Alice + N(0, sqrt(V_noise))

        Returns x_bob, p_bob, noise_budget_dict
        """
        va    = float(np.var(x_alice) + np.var(p_alice)) / 2.0
        noise = self.total_noise(distance_km, va, xi_override)
        T     = noise["T"]
        sigma = np.sqrt(noise["total"])

        x_bob = np.sqrt(T) * x_alice + np.random.normal(0, sigma, x_alice.shape)
        p_bob = np.sqrt(T) * p_alice + np.random.normal(0, sigma, p_alice.shape)

        return x_bob, p_bob, noise
