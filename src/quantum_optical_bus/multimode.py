"""
Multi-mode Gaussian simulation helpers for independent time bins.

This module extends the single-mode API with a per-mode simulation core
without inter-bin coupling. Each mode supports:
- Squeezing (with optional squeezing angle)
- Rotation
- Loss
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Compat patches must be applied before importing Strawberry Fields.
import quantum_optical_bus.compat  # noqa: F401

import strawberryfields as sf
from strawberryfields.ops import LossChannel, Rgate, Sgate

from quantum_optical_bus.units import observed_squeezing_from_cov, sf_cov_to_vacuum05


class MultiModeResult(NamedTuple):
    """Container for independent multi-mode Gaussian simulation outputs."""

    mean_photon: np.ndarray
    var_x: np.ndarray
    var_p: np.ndarray
    observed_sq_db: np.ndarray
    observed_antisq_db: np.ndarray
    wigner: np.ndarray | None
    wigner_mode: int | None
    xvec: np.ndarray | None


def _to_vector(values: float | list[float] | np.ndarray, n_modes: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(n_modes, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D array-like")
    if arr.size == 1:
        return np.full(n_modes, float(arr[0]), dtype=float)
    if arr.size != n_modes:
        raise ValueError(f"{name} has length {arr.size}, expected {n_modes}")
    return arr.astype(float, copy=False)


def _infer_n_modes(
    r: float | list[float] | np.ndarray,
    theta: float | list[float] | np.ndarray,
    eta_loss: float | list[float] | np.ndarray,
    squeeze_theta: float | list[float] | np.ndarray | None,
    n_modes: int | None,
) -> int:
    if n_modes is not None:
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        return int(n_modes)

    lengths = []
    for value in (r, theta, eta_loss, squeeze_theta):
        if value is None:
            continue
        arr = np.asarray(value)
        lengths.append(1 if arr.ndim == 0 else int(arr.size))
    return max(lengths) if lengths else 1


def run_multimode(
    r: float | list[float] | np.ndarray,
    theta: float | list[float] | np.ndarray,
    eta_loss: float | list[float] | np.ndarray,
    *,
    n_modes: int | None = None,
    squeeze_theta: float | list[float] | np.ndarray | None = None,
    xvec: np.ndarray | None = None,
    wigner_mode: int | None = None,
) -> MultiModeResult:
    """Run an independent multi-mode Gaussian circuit.

    Parameters
    ----------
    r : float or array-like
        Per-mode squeezing magnitudes for ``Sgate``.
    theta : float or array-like
        Per-mode phase-space rotation angles (radians) for ``Rgate``.
    eta_loss : float or array-like
        Per-mode transmissivity in ``[0, 1]`` for ``LossChannel``.
    n_modes : int, optional
        Number of modes. If omitted, inferred from longest input vector.
    squeeze_theta : float or array-like, optional
        Per-mode squeezing phase for ``Sgate(r, phi)``. Defaults to 0.
    xvec : np.ndarray, optional
        Quadrature grid for Wigner calculation. Used only if ``wigner_mode``
        is provided. Defaults to ``np.linspace(-4, 4, 120)``.
    wigner_mode : int, optional
        Mode index for optional Wigner evaluation.

    Returns
    -------
    MultiModeResult
        Per-mode metrics in vacuum=0.5 convention and optional Wigner.
    """
    n = _infer_n_modes(r, theta, eta_loss, squeeze_theta, n_modes)

    r_vec = _to_vector(r, n, "r")
    rot_vec = _to_vector(theta, n, "theta")
    eta_vec = _to_vector(eta_loss, n, "eta_loss")
    sq_phase_vec = _to_vector(0.0 if squeeze_theta is None else squeeze_theta, n, "squeeze_theta")

    if np.any((eta_vec < 0.0) | (eta_vec > 1.0)):
        raise ValueError("eta_loss entries must be within [0, 1]")

    if wigner_mode is not None and (wigner_mode < 0 or wigner_mode >= n):
        raise ValueError(f"wigner_mode must be in [0, {n - 1}]")

    grid = None
    if wigner_mode is not None:
        grid = np.linspace(-4.0, 4.0, 120) if xvec is None else np.asarray(xvec, dtype=float)
        if grid.ndim != 1:
            raise ValueError("xvec must be a 1-D array")

    prog = sf.Program(n)
    with prog.context as q:
        for idx in range(n):
            if r_vec[idx] != 0.0:
                Sgate(r_vec[idx], sq_phase_vec[idx]) | q[idx]
            if rot_vec[idx] != 0.0:
                Rgate(rot_vec[idx]) | q[idx]
            if eta_vec[idx] < 1.0:
                LossChannel(eta_vec[idx]) | q[idx]

    state = sf.Engine("gaussian").run(prog).state

    cov = sf_cov_to_vacuum05(state.cov())

    mean_photon = np.zeros(n, dtype=float)
    var_x = np.zeros(n, dtype=float)
    var_p = np.zeros(n, dtype=float)
    observed_sq_db = np.zeros(n, dtype=float)
    observed_antisq_db = np.zeros(n, dtype=float)

    for idx in range(n):
        mean_photon[idx] = float(state.mean_photon(idx)[0])
        x_idx = idx
        p_idx = n + idx
        block = cov[np.ix_([x_idx, p_idx], [x_idx, p_idx])]
        var_x[idx] = float(cov[x_idx, x_idx])
        var_p[idx] = float(cov[p_idx, p_idx])
        observed_sq_db[idx], observed_antisq_db[idx] = observed_squeezing_from_cov(block)

    wigner = None
    if wigner_mode is not None and grid is not None:
        wigner = state.wigner(int(wigner_mode), grid, grid)

    return MultiModeResult(
        mean_photon=mean_photon,
        var_x=var_x,
        var_p=var_p,
        observed_sq_db=observed_sq_db,
        observed_antisq_db=observed_antisq_db,
        wigner=wigner,
        wigner_mode=wigner_mode,
        xvec=grid,
    )
