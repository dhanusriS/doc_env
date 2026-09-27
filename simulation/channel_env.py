"""
simulation/channel_env.py
=========================
AR(1) time-varying channel environment for temporal simulation.

NO BUGS in this module — included unchanged from v1.
Simulates realistic fiber fluctuations using correlated AR(1) dynamics.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """AR(1) channel environment parameters."""
    mu_T:     float = 0.12     # Mean transmittance (~22 km equivalent)
    mu_xi:    float = 0.010    # Mean excess noise (SNU)
    rho_T:    float = 0.95     # AR(1) correlation for T
    rho_xi:   float = 0.90     # AR(1) correlation for xi
    sigma_T:  float = 0.015    # Innovation std for T
    sigma_xi: float = 0.004    # Innovation std for xi
    T_min:    float = 0.005
    T_max:    float = 0.80
    xi_min:   float = 0.001
    xi_max:   float = 0.12


class ChannelEnvironment:
    """
    Temporally correlated fiber channel (AR(1) process).

    T_{t+1}  = rho_T  * T_t  + (1-rho_T)  * mu_T  + innovation
    xi_{t+1} = rho_xi * xi_t + (1-rho_xi) * mu_xi + innovation

    Supports step-change injection to simulate WDM channel switch-on.
    """

    def __init__(self, cfg: EnvConfig | None = None, seed: int | None = None):
        self.cfg = cfg or EnvConfig()
        if seed is not None:
            np.random.seed(seed)
        self.T          = self.cfg.mu_T
        self.xi         = self.cfg.mu_xi
        self.t          = 0
        self.T_history:  list[float] = []
        self.xi_history: list[float] = []

    def step(self) -> tuple[float, float]:
        """Advance one block. Returns (T_t, xi_t)."""
        cfg = self.cfg
        innov_T  = np.random.normal(0, cfg.sigma_T  * np.sqrt(1 - cfg.rho_T  ** 2))
        innov_xi = np.random.normal(0, cfg.sigma_xi * np.sqrt(1 - cfg.rho_xi ** 2))
        self.T   = float(np.clip(cfg.rho_T  * self.T  + (1 - cfg.rho_T)  * cfg.mu_T  + innov_T,  cfg.T_min,  cfg.T_max))
        self.xi  = float(np.clip(cfg.rho_xi * self.xi + (1 - cfg.rho_xi) * cfg.mu_xi + innov_xi, cfg.xi_min, cfg.xi_max))
        self.t  += 1
        self.T_history.append(self.T)
        self.xi_history.append(self.xi)
        return self.T, self.xi

    def inject_step_change(self, delta_xi: float = 0.015, delta_T: float = 0.0):
        """Inject sudden step change (e.g. WDM channel switched on)."""
        self.xi = float(np.clip(self.xi + delta_xi, self.cfg.xi_min, self.cfg.xi_max))
        self.T  = float(np.clip(self.T  + delta_T,  self.cfg.T_min,  self.cfg.T_max))

    def reset(self, T: float | None = None, xi: float | None = None):
        """Reset to initial conditions."""
        self.T  = T  or self.cfg.mu_T
        self.xi = xi or self.cfg.mu_xi
        self.t  = 0
        self.T_history.clear()
        self.xi_history.clear()

    def run_trajectory(
        self,
        n_blocks:       int,
        step_change_at: int | None = None,
        delta_xi:       float      = 0.015,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate full trajectory of n_blocks (T, xi) pairs."""
        self.reset()
        T_arr  = np.zeros(n_blocks)
        xi_arr = np.zeros(n_blocks)
        for i in range(n_blocks):
            if step_change_at is not None and i == step_change_at:
                self.inject_step_change(delta_xi=delta_xi)
            T_arr[i], xi_arr[i] = self.step()
        return T_arr, xi_arr
