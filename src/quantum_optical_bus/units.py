"""
Shared unit conversions and covariance post-processing helpers.
"""

from __future__ import annotations

import numpy as np

VACUUM_VAR_05 = 0.5
_SF_COV_SCALE_TO_VACUUM05 = 0.5


def db_to_eta(loss_db: float | np.ndarray) -> float | np.ndarray:
    """Convert loss in dB to linear transmissivity eta."""
    arr = np.asarray(loss_db, dtype=float)
    eta = np.power(10.0, -arr / 10.0)
    return float(eta) if eta.ndim == 0 else eta


def eta_to_db(eta: float | np.ndarray) -> float | np.ndarray:
    """Convert linear transmissivity eta to loss in dB."""
    arr = np.asarray(eta, dtype=float)
    if np.any(arr < 0.0):
        raise ValueError("eta must be non-negative")
    with np.errstate(divide="ignore"):
        loss_db = -10.0 * np.log10(arr)
    return float(loss_db) if loss_db.ndim == 0 else loss_db


def sf_cov_to_vacuum05(cov: np.ndarray) -> np.ndarray:
    """Convert Strawberry Fields covariance (hbar=2) to vacuum=0.5 convention."""
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2:
        raise ValueError("cov must be a 2-D array")
    return arr * _SF_COV_SCALE_TO_VACUUM05


def observed_squeezing_from_cov(cov_vacuum05: np.ndarray) -> tuple[float, float]:
    """Return observed squeezing / anti-squeezing (dB) from covariance eigenvalues."""
    arr = np.asarray(cov_vacuum05, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("cov_vacuum05 must be a square 2-D array")

    eigvals = np.linalg.eigvalsh(arr)
    vmin = float(eigvals[0])
    vmax = float(eigvals[-1])
    observed_sq_db = float(-10.0 * np.log10(vmin / VACUUM_VAR_05)) if vmin > 0.0 else 0.0
    observed_antisq_db = float(10.0 * np.log10(vmax / VACUUM_VAR_05)) if vmax > 0.0 else 0.0
    return observed_sq_db, observed_antisq_db
