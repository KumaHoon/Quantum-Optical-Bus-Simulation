"""
Quantum simulation helpers — shared single-mode Gaussian circuits.

Provides ``run_single_mode`` which executes a squeezed-state circuit
(optional rotation and loss) on the Strawberry Fields Gaussian backend
and returns all derived metrics in a ``QuantumResult`` namedtuple.
"""

from typing import NamedTuple

import numpy as np

# Compat patches must be applied before importing Strawberry Fields.
import quantum_optical_bus.compat  # noqa: F401

import strawberryfields as sf
from strawberryfields.ops import LossChannel, Rgate, Sgate

from quantum_optical_bus.units import observed_squeezing_from_cov, sf_cov_to_vacuum05


class QuantumResult(NamedTuple):
    """Container for single-mode Gaussian simulation outputs."""

    W: np.ndarray
    """2-D Wigner function evaluated on *xvec × xvec*."""

    mean_photon: float
    """Mean photon number ⟨n̂⟩."""

    var_x: float
    """Position-quadrature variance (normalized so vacuum = 0.5)."""

    var_p: float
    """Momentum-quadrature variance (normalized so vacuum = 0.5)."""

    observed_sq_db: float
    """Observed squeezing (post-loss) in dB, from covariance eigenvalues."""

    observed_antisq_db: float
    """Observed anti-squeezing in dB, from covariance eigenvalues."""


def run_single_mode(
    r: float,
    theta: float,
    eta_loss: float,
    xvec: np.ndarray,
) -> QuantumResult:
    """Run a single-mode Gaussian circuit and return state metrics.

    Parameters
    ----------
    r : float
        Squeezing parameter for the ``Sgate``.
    theta : float
        Phase-space rotation angle (radians) for the ``Rgate``.
    eta_loss : float
        Channel transmissivity in ``[0, 1]``.  Set to ``1.0`` for no loss.
    xvec : np.ndarray
        1-D array of quadrature values for Wigner function evaluation.

    Returns
    -------
    QuantumResult
        Named tuple with Wigner function, photon number, variances,
        and observed squeezing / anti-squeezing in dB.
    """
    prog = sf.Program(1)
    with prog.context as q:
        if r > 0:
            Sgate(r) | q[0]
        if theta != 0:
            Rgate(theta) | q[0]
        if eta_loss < 1.0:
            LossChannel(eta_loss) | q[0]

    state = sf.Engine("gaussian").run(prog).state

    W = state.wigner(0, xvec, xvec)
    mean_n = state.mean_photon(0)[0]
    cov = sf_cov_to_vacuum05(state.cov())
    var_x = float(cov[0, 0])
    var_p = float(cov[1, 1])
    observed_sq_db, observed_antisq_db = observed_squeezing_from_cov(cov)

    return QuantumResult(
        W=W,
        mean_photon=mean_n,
        var_x=var_x,
        var_p=var_p,
        observed_sq_db=observed_sq_db,
        observed_antisq_db=observed_antisq_db,
    )
