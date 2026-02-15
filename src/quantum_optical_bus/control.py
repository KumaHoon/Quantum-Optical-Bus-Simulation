"""
Drift and latency-aware feedback simulation for TDM control loops.
"""

from __future__ import annotations

from typing import Callable, Any

import numpy as np


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def simulate_phase_drift(
    T: int,
    step_sigma: float,
    *,
    drift_rate: float = 0.0,
    initial_phase: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate phase drift as a random walk with optional deterministic trend."""
    if T < 1:
        raise ValueError("T must be >= 1")
    if step_sigma < 0:
        raise ValueError("step_sigma must be >= 0")

    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=drift_rate, scale=step_sigma, size=max(T - 1, 0))
    phase = np.empty(T, dtype=float)
    phase[0] = float(initial_phase)
    if T > 1:
        phase[1:] = phase[0] + np.cumsum(steps)
    return _wrap_phase(phase)


def apply_feedback_with_latency(
    *,
    latency_steps: int,
    true_phase: np.ndarray | None = None,
    T: int | None = None,
    step_sigma: float = 0.01,
    drift_rate: float = 0.0,
    estimator: Callable[..., float] | None = None,
    controller: Callable[..., float] | None = None,
    measurement_sigma: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Apply delayed phase feedback and report residual error statistics.

    Notes
    -----
    This MVP model applies control commands generated at step *t* after
    ``latency_steps`` delay, approximating a TDM loop feedback constraint.
    """
    if latency_steps < 0:
        raise ValueError("latency_steps must be >= 0")

    if true_phase is None:
        if T is None:
            raise ValueError("Provide true_phase or T")
        phase = simulate_phase_drift(T=T, step_sigma=step_sigma, drift_rate=drift_rate, seed=seed)
    else:
        phase = np.asarray(true_phase, dtype=float)
        if phase.ndim != 1:
            raise ValueError("true_phase must be 1-D")
        phase = _wrap_phase(phase)
        T = int(phase.size)

    rng = np.random.default_rng(seed)

    if estimator is None:

        def estimator_fn(phase_value: float, _t: int, _rng: np.random.Generator) -> float:
            return float(phase_value + _rng.normal(0.0, measurement_sigma))
    else:

        def estimator_fn(phase_value: float, t: int, _rng: np.random.Generator) -> float:
            return float(estimator(phase_value, t=t, rng=_rng))

    if controller is None:

        def controller_fn(estimated_phase: float, _t: int) -> float:
            return float(estimated_phase)
    else:

        def controller_fn(estimated_phase: float, t: int) -> float:
            return float(controller(estimated_phase, t=t))

    # Command scheduled at each time index (absolute phase to cancel).
    scheduled_command = np.zeros(T, dtype=float)

    for t in range(T):
        measured = estimator_fn(float(phase[t]), t, rng)
        command = controller_fn(measured, t)
        apply_t = t + latency_steps
        if apply_t < T:
            scheduled_command[apply_t] = command

    residual = _wrap_phase(phase - scheduled_command)
    retention = np.exp(-(residual**2))

    return {
        "true_phase": phase,
        "applied_command": scheduled_command,
        "residual_phase_error": residual,
        "squeezing_retention_proxy": retention,
        "rms_residual_phase_error": float(np.sqrt(np.mean(residual**2))),
        "mean_retention_proxy": float(np.mean(retention)),
        "latency_steps": int(latency_steps),
    }
