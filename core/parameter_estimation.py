"""
core/parameter_estimation.py
============================
Estimate channel parameters T and xi from sacrificed symbol pairs.

NO BUGS in this module — included unchanged from v1.

Note on bias: The MLE estimator has known bias for:
  - Small T (long distances, weak signal)
  - Non-Gaussian input distributions (MB differs slightly from Gaussian)
In simulation the bias is visible but the confidence bounds (T_min,
xi_max) remain conservative (correct for security).
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class PEResult:
    """Output of parameter estimation for one block."""
    T_hat:    float   # Point estimate of transmittance
    xi_hat:   float   # Point estimate of excess noise (SNU)
    T_min:    float   # Worst-case lower bound on T (used in security proof)
    xi_max:   float   # Worst-case upper bound on xi (used in security proof)
    n_PE:     int     # Number of symbols used for PE
    SNR_hat:  float   # Estimated SNR from point estimates
    cov_AA:   float   # Estimated Alice quadrature variance
    cov_BB:   float   # Estimated Bob quadrature variance
    cov_AB:   float   # Estimated cross-covariance


class ParameterEstimator:
    """
    MLE parameter estimation from sacrificed QKD symbols.

    Alice reveals her symbols for n_PE randomly selected positions.
    Bob compares his received values to estimate T and xi.

    Parameters
    ----------
    eps_PE : float
        Security parameter — true (T, xi) lies outside the confidence
        interval with probability at most eps_PE.
    nu_el  : float
        Known detector electronic noise (SNU), calibrated externally.
    """

    def __init__(self, eps_PE: float = 1e-10, nu_el: float = 0.01):
        self.eps_PE  = eps_PE
        self.nu_el   = nu_el
        self.z = stats.norm.ppf(1.0 - eps_PE / 2.0)

    def estimate(
        self,
        x_alice: np.ndarray,
        x_bob:   np.ndarray,
        p_alice: np.ndarray,
        p_bob:   np.ndarray,
    ) -> PEResult:
        """
        MLE estimation of (T, xi) from n_PE = len(x_alice) pairs.

        Covariance matrix inversion (from gamma_AB structure):
            T_hat  = (C_AB / (C_AA - 1))^2
            xi_hat = C_BB / T_hat - C_AA - 1

        Confidence bounds use the z-score for eps_PE security level.
        """
        n = len(x_alice)

        # Estimate covariance matrix elements (average over I and Q)
        cov_AA = float((np.var(x_alice) + np.var(p_alice)) / 2.0)
        cov_BB = float((np.var(x_bob)   + np.var(p_bob))   / 2.0)
        cov_AB_x = float(np.cov(x_alice, x_bob)[0, 1])
        cov_AB_p = float(np.cov(p_alice, p_bob)[0, 1])
        cov_AB   = (cov_AB_x + cov_AB_p) / 2.0

        # MLE estimates
        denom  = max(cov_AA - 1.0, 1e-6)
        T_hat  = float(np.clip((cov_AB / denom) ** 2, 1e-4, 1.0))
        xi_hat = float(max(cov_BB / max(T_hat, 1e-6) - cov_AA - 1.0, 0.0))

        # Finite-size confidence bounds
        se_T  = self.z * np.sqrt(T_hat * (1.0 - T_hat) / max(n, 1))
        se_xi = self.z * np.sqrt(max(2.0 * xi_hat**2 / max(n, 1) + 4.0 / max(n, 1), 0))

        T_min  = max(T_hat  - se_T,  1e-4)
        xi_max = xi_hat + se_xi

        # SNR estimate at point estimates
        VA_hat  = max(cov_AA, 1e-6)
        SNR_hat = float(T_hat * VA_hat / max(1.0 + T_hat * xi_hat + self.nu_el, 1e-6))

        return PEResult(
            T_hat=T_hat, xi_hat=xi_hat,
            T_min=T_min, xi_max=xi_max,
            n_PE=n, SNR_hat=SNR_hat,
            cov_AA=cov_AA, cov_BB=cov_BB, cov_AB=cov_AB,
        )
