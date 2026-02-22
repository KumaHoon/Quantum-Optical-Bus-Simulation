"""
Drift and latency-aware feedback simulation for TDM control loops.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Any

import numpy as np

from .estimation import fit_eta_and_loss


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _quantize_to_bits(
    values: np.ndarray,
    bits: int,
    *,
    low: float = -np.pi,
    high: float = np.pi,
) -> np.ndarray:
    if bits <= 0:
        raise ValueError("bits must be positive")
    if bits >= 16:
        return values.astype(float, copy=True)
    if not np.isfinite(low) or not np.isfinite(high) or not (low < high):
        raise ValueError("low/high must be finite and low < high")

    levels = 1 << bits
    clipped = np.clip(values, low, high)
    step = (high - low) / (levels - 1)
    return np.round((clipped - low) / step) * step + low


def _infer_measurement_count(data: dict[str, Any] | str | Path) -> int | None:
    if isinstance(data, dict):
        pump = data.get("pump_power_mw")
        if pump is None:
            return None
        return int(np.asarray(pump).size)

    if isinstance(data, (str, Path)):
        count = 0
        with Path(data).open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for _ in reader:
                count += 1
        return count

    return None


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


def run_measurement_to_control_pipeline(
    measurement_data: dict[str, Any] | str | Path,
    *,
    model: str = "auto",
    latency_steps: int = 2,
    n_steps: int | None = None,
    measurement_sigma: float = 0.0,
    estimator_bits: int | None = None,
    seed: int | None = None,
    step_sigma: float | None = None,
    drift_rate: float | None = None,
) -> dict[str, Any]:
    """Run a measurement → estimation → control chain with a synthetic plant path.

    Parameters
    ----------
    measurement_data
        Calibration table (dict-like) or path to a CSV file accepted by
        ``fit_eta_and_loss``.
    model
        Estimation model passed through to ``fit_eta_and_loss``.
    latency_steps
        Feedback latency in steps.
    n_steps
        Drift trajectory length. If omitted, derive from measurement length when available.
    measurement_sigma
        Additive measurement noise standard deviation in radians.
    estimator_bits
        Optional scalar quantizer applied to the phase estimate.
    seed
        RNG seed for drift and noise (reused deterministically).
    step_sigma
        Phase random-walk step scale. If omitted, inferred from fitted loss.
    drift_rate
        Phase deterministic slope. If omitted, inferred from fitted loss.
    """
    eta_hat, loss_hat, diagnostics = fit_eta_and_loss(measurement_data, model=model)

    if n_steps is None:
        inferred_steps = _infer_measurement_count(measurement_data)
        n_steps = inferred_steps if inferred_steps is not None else 220

    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    if step_sigma is None:
        step_sigma = 0.010 + 0.0010 * max(loss_hat, 0.0)
    if drift_rate is None:
        drift_rate = 0.0009 + 0.0002 * max(loss_hat, 0.0)

    phase = simulate_phase_drift(
        T=n_steps,
        step_sigma=step_sigma,
        drift_rate=drift_rate,
        seed=seed,
    )

    estimator = None
    if estimator_bits is not None:

        def estimator(phase_value: float, t: int, rng: np.random.Generator) -> float:
            measured = float(phase_value + rng.normal(0.0, measurement_sigma))
            return float(_quantize_to_bits(np.array([measured]), bits=estimator_bits)[0])

    control_result = apply_feedback_with_latency(
        latency_steps=latency_steps,
        true_phase=phase,
        estimator=estimator,
        measurement_sigma=measurement_sigma,
        seed=None if seed is None else seed + 100,
    )

    return {
        "eta_hat": eta_hat,
        "loss_db_hat": loss_hat,
        "fit_diagnostics": diagnostics,
        "phase_trace": phase,
        "fit_model": diagnostics.get("model", model),
        "control_steps": n_steps,
        "control_result": control_result,
    }


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
