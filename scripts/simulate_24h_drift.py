"""Simulate drift and auto-calibration recovery for a synthetic drift process."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DriftProfile:
    """Deterministic trajectory for drifted parameters."""

    time_hours: np.ndarray
    phase_rad: np.ndarray
    loss_db: np.ndarray
    squeezing_db: np.ndarray


@dataclass(frozen=True)
class RecoveryTrace:
    """Output trace from drift recovery simulation."""

    phase_rad: np.ndarray
    estimated_phase: np.ndarray
    measured_phase: np.ndarray
    residual_rad: np.ndarray
    command: np.ndarray
    controller_gain: np.ndarray
    loss_db: np.ndarray
    squeezing_db: np.ndarray
    time_hours: np.ndarray


def _wrap_phase(phase: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(phase) + np.pi) % (2.0 * np.pi) - np.pi


def generate_drift_profile(
    *,
    total_hours: float,
    steps_per_hour: int,
    seed: int,
    phase_drift_start: float,
    phase_drift_end: float,
    loss_drift_db_per_hour: float,
    squeezing_drift_db_per_hour: float,
) -> DriftProfile:
    """Generate a deterministic drift trajectory."""
    if total_hours <= 0:
        raise ValueError("total_hours must be > 0")
    if steps_per_hour < 1:
        raise ValueError("steps_per_hour must be >= 1")

    steps = int(round(total_hours * steps_per_hour))
    rng = np.random.default_rng(seed)
    time_hours = np.arange(steps) / float(steps_per_hour)

    phase = np.zeros(steps, dtype=float)
    loss_db = np.full(steps, 1.8, dtype=float)
    squeezing_db = np.full(steps, 10.0, dtype=float)

    phase_drift = np.interp(
        time_hours,
        [0.0, total_hours],
        [phase_drift_start, phase_drift_end],
    )

    for i in range(1, steps):
        phase[i] = phase[i - 1] + phase_drift[i] / steps_per_hour
        phase[i] += rng.normal(loc=0.0, scale=0.0006)
        loss_db[i] = loss_db[i - 1] + loss_drift_db_per_hour / steps_per_hour
        loss_db[i] += rng.normal(loc=0.0, scale=0.003)
        squeezing_db[i] = squeezing_db[i - 1] + squeezing_drift_db_per_hour / steps_per_hour
        squeezing_db[i] += rng.normal(loc=0.0, scale=0.006)

    loss_db = np.clip(loss_db, 1.1, 2.8)
    squeezing_db = np.clip(squeezing_db, 8.0, 12.0)
    return DriftProfile(
        time_hours=time_hours,
        phase_rad=_wrap_phase(phase),
        loss_db=loss_db,
        squeezing_db=squeezing_db,
    )


def run_drift_recovery(
    profile: DriftProfile,
    *,
    seed: int,
    measurement_sigma: float = 0.015,
    estimator_alpha: float = 0.18,
    controller_gain_init: float = 1.0,
    controller_gain_min: float = 0.35,
    controller_gain_max: float = 1.6,
    actuation_latency_steps: int = 2,
    controller_update_interval_steps: int = 90,
    controller_update_gain_step: float = 0.06,
    integral_gain: float = 0.08,
) -> RecoveryTrace:
    """Run a simple EMA estimator and gain-tuning controller loop."""
    if not 0 < estimator_alpha <= 1:
        raise ValueError("estimator_alpha must be in (0, 1]")
    if controller_gain_init <= 0:
        raise ValueError("controller_gain_init must be > 0")
    if controller_update_interval_steps < 1:
        raise ValueError("controller_update_interval_steps must be >= 1")

    rng = np.random.default_rng(seed)
    phase = profile.phase_rad
    n_steps = phase.size

    measured = np.empty_like(phase)
    estimate = np.empty_like(phase)
    residual = np.empty_like(phase)
    command = np.empty_like(phase)
    gain = np.empty_like(phase)
    command_pipeline = np.zeros(actuation_latency_steps + 1, dtype=float)

    gain_value = float(controller_gain_init)
    integral = 0.0
    target_rms = measurement_sigma * np.sqrt(0.95)

    estimate[0] = float(_wrap_phase(phase[0] + rng.normal(0.0, measurement_sigma)))
    measured[0] = estimate[0]
    command_pipeline[-1] = 0.0
    applied_command = command_pipeline[0]
    command_pipeline = np.roll(command_pipeline, -1)
    command_pipeline[-1] = 0.0
    command[0] = applied_command
    residual[0] = _wrap_phase(phase[0] - command[0])
    gain[0] = gain_value

    for t in range(1, n_steps):
        measured[t] = phase[t] + rng.normal(0.0, measurement_sigma)
        measured[t] = float(_wrap_phase(measured[t]))
        estimate[t] = float(
            _wrap_phase(estimator_alpha * measured[t] + (1.0 - estimator_alpha) * estimate[t - 1])
        )

        command_now = float(_wrap_phase(gain_value * estimate[t] + integral_gain * integral))
        command_pipeline[-1] = command_now
        applied = command_pipeline[0]
        command_pipeline = np.roll(command_pipeline, -1)
        command_pipeline[-1] = 0.0
        command[t] = float(applied)
        residual[t] = float(_wrap_phase(phase[t] - command[t]))
        integral = float(0.96 * integral + (1.0 - 0.96) * estimate[t])
        gain[t] = gain_value

        if t % controller_update_interval_steps == 0:
            window = residual[max(0, t - controller_update_interval_steps + 1) : t + 1]
            recent_rms = float(np.sqrt(np.mean(window**2)))
            if recent_rms > target_rms * 1.8:
                gain_value = min(
                    controller_gain_max, gain_value * (1 + controller_update_gain_step)
                )
            elif recent_rms < target_rms * 1.2:
                gain_value = max(
                    controller_gain_min, gain_value * (1 - controller_update_gain_step * 0.4)
                )
            gain[t] = gain_value

    return RecoveryTrace(
        phase_rad=phase,
        measured_phase=measured,
        estimated_phase=estimate,
        residual_rad=residual,
        command=command,
        controller_gain=gain,
        loss_db=profile.loss_db,
        squeezing_db=profile.squeezing_db,
        time_hours=profile.time_hours,
    )


def plot_recovery(trace: RecoveryTrace, output_path: Path) -> None:
    """Create the stability/recovery artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(trace.time_hours, trace.phase_rad, label="true phase")
    axes[0].plot(trace.time_hours, trace.estimated_phase, label="estimated phase")
    axes[0].plot(trace.time_hours, trace.command, label="applied correction")
    axes[0].set_title("24h Drift Recovery: Drift, Estimator, Controller")
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Phase [rad]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(trace.time_hours, trace.residual_rad)
    axes[1].set_title("Residual phase after correction")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Residual [rad]")
    axes[1].grid(alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.4)

    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    ax_loss.plot(trace.time_hours, trace.loss_db, color="tab:blue", label="loss [dB]")
    ax_loss.plot(trace.time_hours, trace.squeezing_db, color="tab:orange", label="squeezing [dB]")
    ax_loss.set_xlabel("Time [h]")
    ax_loss.set_ylabel("Loss / squeezing")
    ax_loss.set_title("Recovered drift, measured loss/squeezing, adaptive gain")

    ax_gain.plot(
        trace.time_hours,
        trace.controller_gain,
        color="tab:green",
        linestyle="--",
        label="controller gain",
    )
    ax_gain.set_ylabel("Controller gain")

    ax_loss.grid(alpha=0.25)
    ax_loss.legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_24h_drift(
    *,
    output_path: Path,
    total_hours: float = 24.0,
    steps_per_hour: int = 60,
    seed: int = 17,
) -> Path:
    """Run the full drift/estimation/recovery pipeline and save the artifact."""
    profile = generate_drift_profile(
        total_hours=total_hours,
        steps_per_hour=steps_per_hour,
        seed=seed,
        phase_drift_start=0.0022,
        phase_drift_end=0.0075,
        loss_drift_db_per_hour=0.012,
        squeezing_drift_db_per_hour=-0.04,
    )
    trace = run_drift_recovery(
        profile,
        seed=seed + 111,
        measurement_sigma=0.02,
        estimator_alpha=0.22,
        controller_gain_init=0.95,
        controller_update_interval_steps=steps_per_hour,
    )
    plot_recovery(trace, output_path=output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible 24h drift recovery sweep.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "drift_recovery.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--total-hours", type=float, default=24.0)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run_24h_drift(
        output_path=Path(args.output),
        total_hours=args.total_hours,
        steps_per_hour=args.steps_per_hour,
        seed=args.seed,
    )
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
