"""
Layer 2 — Interface Layer

Maps physical pump power (mW) to the quantum squeezing parameter *r*
using a phenomenological model for Lithium Niobate waveguides.
"""

import numpy as np

# Phenomenological coupling efficiency.
# Tuned so that 100 mW → r ≈ 1.0.
_COUPLING_EFFICIENCY = 0.1


def calculate_squeezing(pump_power_mw: float | np.ndarray) -> float | np.ndarray:
    """
    Convert pump power to squeezing parameter.

    Parameters
    ----------
    pump_power_mw : float or np.ndarray
        Pump power in milliwatts (≥ 0).

    Returns
    -------
    r : float or np.ndarray
        Squeezing parameter for a Strawberry Fields ``Sgate``.
    """
    pump = np.asarray(pump_power_mw, dtype=float)
    r = _COUPLING_EFFICIENCY * np.sqrt(pump)
    return float(r) if r.ndim == 0 else r
