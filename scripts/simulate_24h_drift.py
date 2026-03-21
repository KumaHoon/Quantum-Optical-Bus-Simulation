"""Simulate drift and auto-calibration recovery for a synthetic drift process."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_optical_bus.viz_style_ieee import (
    SERIES_BLUE,
    SERIES_ORANGE,
    SERIES_TEAL,
    apply_review_layout,
    set_tab_title,
    set_review_axis,
    save_ieee,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
try:
    from asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs
except ModuleNotFoundError:
    from scripts.asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs
try:
    from figstyle import (
        apply_style,
        canonical_canvas_inches,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_meta,
    )
except ModuleNotFoundError:
    from scripts.figstyle import (
        apply_style,
        canonical_canvas_inches,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_meta,
    )


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


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


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


def _resolve_output_targets(output_path: Path, profile: str) -> list[tuple[Path, str]]:
    output_path = Path(output_path)
    if profile == "web":
        return [(output_path, "web")]
    if profile == "paper":
        paper_root = (
            output_path.parent / "paper"
            if output_path.parent.name != "paper"
            else output_path.parent
        )
        return [(paper_root / output_path.name, "paper")]
    return [
        (target, "web" if target.parent.name == "web" else "paper")
        for target in dict.fromkeys(resolve_outputs(output_path, profile))
    ]


def plot_recovery(
    trace: RecoveryTrace,
    output_path: Path,
    *,
    profile: str = "web",
    seed: int = 17,
) -> None:
    apply_style(
        profile,
        base_font_size=11 if profile == "paper" else 10,
        tick_font_size=10 if profile == "paper" else 9,
        dpi=canonical_dpi(profile),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=canonical_canvas_inches(profile),
    )

    axes[0].plot(trace.time_hours, trace.phase_rad, label="true phase", color=SERIES_ORANGE)
    axes[0].plot(trace.time_hours, trace.estimated_phase, label="estimate", color=SERIES_BLUE)
    axes[0].plot(trace.time_hours, trace.command, label="applied correction", color=SERIES_TEAL)
    set_review_axis(
        axes[0],
        title="24h Drift Recovery: Drift, Estimator, Controller",
        xlabel="Time (hours)",
        ylabel="Phase (rad)",
        integer_ticks=True,
    )
    axes[0].legend(loc="upper right", frameon=False, fontsize=7, ncol=1)
    axes[0].grid(alpha=0.25)

    axes[1].plot(trace.time_hours, trace.residual_rad)
    set_review_axis(
        axes[1],
        title="Residual phase after correction",
        xlabel="Time (hours)",
        ylabel="Residual phase (rad)",
        integer_ticks=False,
    )
    axes[1].grid(alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)

    ax_loss = axes[2]
    ax_gain = ax_loss.twinx()
    ax_loss.plot(
        trace.time_hours,
        trace.loss_db,
        color=SERIES_BLUE,
        label="loss (dB)",
    )
    ax_loss.plot(
        trace.time_hours,
        trace.squeezing_db,
        color=SERIES_ORANGE,
        label="squeezing (dB)",
    )
    ax_loss.set_ylim(
        np.min([np.min(trace.loss_db), np.min(trace.squeezing_db)]) - 0.2,
        np.max([np.max(trace.loss_db), np.max(trace.squeezing_db)]) + 0.2,
    )

    ax_loss.set_yticks(
        np.linspace(
            np.floor(ax_loss.get_ylim()[0] * 10) / 10,
            np.ceil(ax_loss.get_ylim()[1] * 10) / 10,
            num=6,
        )
    )
    set_review_axis(
        ax_loss,
        title="Recovered drift, measured loss/squeezing, adaptive gain",
        xlabel="Time (hours)",
        ylabel="Loss / squeezing (dB)",
        integer_ticks=True,
    )
    ax_loss.tick_params(axis="y", colors=SERIES_BLUE)
    ax_loss.grid(alpha=0.25)

    ax_gain.plot(
        trace.time_hours,
        trace.controller_gain,
        color=SERIES_TEAL,
        linestyle="--",
        label="controller gain",
    )
    ax_gain.set_ylabel("Controller gain (unitless)")
    ax_gain.tick_params(axis="y", colors=SERIES_TEAL)

    handles = [*ax_loss.get_lines(), *ax_gain.get_lines()]
    labels = [line.get_label() for line in handles]
    ax_loss.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.8, ncol=1)
    ax_loss.text(
        0.01,
        1.04,
        "Controller: EMA estimator + adaptive gain scheduler.",
        transform=ax_loss.transAxes,
        fontsize=8,
        color=SERIES_BLUE,
    )

    set_tab_title(fig, "Phase Drift Recovery (24 h simulation)", mode=profile)
    apply_review_layout(
        fig,
        mode=profile,
        left=0.08,
        right=0.97,
        bottom=0.07,
        top=0.90,
        wspace=0.20,
        hspace=0.25,
    )
    for target, target_profile in _resolve_output_targets(output_path, profile):
        dpi = canonical_dpi(target_profile)
        canvas_px = canonical_canvas_px(target_profile)
        save_ieee(fig, target, dpi=dpi, skip_tight_layout=True)
        write_figure_meta(
            target,
            figure_id=target.stem,
            profile=target_profile,
            generator_script="scripts/simulate_24h_drift.py",
            generator_args=(
                f"--output-dir={output_path.parent}",
                f"--profile={target_profile}",
            ),
            labels={
                "title": "24h Drift Recovery",
                "xlabel": "Time (hours)",
                "ylabel": "Phase / Loss / Squeezing (rad / dB)",
            },
            units={
                "time": "h",
                "phase": "rad",
                "loss": "dB",
                "squeezing": "dB",
            },
            notes=(
                "Closed-loop drift recovery trace with EMA estimation and adaptive controller gain."
            ),
            seed=seed,
            dpi=dpi,
            canvas_px=canvas_px,
            data_payload={
                "time_h": trace.time_hours,
                "true_phase_rad": trace.phase_rad,
                "estimated_phase_rad": trace.estimated_phase,
                "measured_phase_rad": trace.measured_phase,
                "residual_rad": trace.residual_rad,
                "command_rad": trace.command,
                "controller_gain": trace.controller_gain,
                "loss_db": trace.loss_db,
                "squeezing_db": trace.squeezing_db,
            },
        )
    plt.close(fig)


def run_24h_drift(
    *,
    output_path: Path,
    total_hours: float = 24.0,
    steps_per_hour: int = 60,
    seed: int = 17,
    profile: str = "web",
) -> Path:
    """Run the full drift/estimation/recovery pipeline and save the artifact."""
    profiles = normalize_profiles(profile)

    profile_data = generate_drift_profile(
        total_hours=total_hours,
        steps_per_hour=steps_per_hour,
        seed=seed,
        phase_drift_start=0.0022,
        phase_drift_end=0.0075,
        loss_drift_db_per_hour=0.012,
        squeezing_drift_db_per_hour=-0.04,
    )
    trace = run_drift_recovery(
        profile_data,
        seed=seed + 111,
        measurement_sigma=0.02,
        estimator_alpha=0.22,
        controller_gain_init=0.95,
        controller_update_interval_steps=steps_per_hour,
        integral_gain=0.0,
    )
    for profile_name in profiles:
        plot_recovery(trace, output_path=output_path, profile=profile_name, seed=seed)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible 24h drift recovery sweep.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory where drift artifact is written.",
    )
    parser.add_argument(
        "--output",
        default="drift_recovery.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--profile",
        default="web",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    parser.add_argument("--total-hours", type=float, default=24.0)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_arg = Path(args.output)
    output_path = (
        args.output_dir / output_arg.name if output_arg.parent == Path(".") else output_arg
    )
    path = run_24h_drift(
        output_path=output_path,
        total_hours=args.total_hours,
        steps_per_hour=args.steps_per_hour,
        seed=args.seed,
        profile=args.profile,
    )
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
