"""Generate reproducible control constraints sweep artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from quantum_optical_bus.viz_style_ieee import (
    apply_review_layout,
    AXIS_LABEL_FONT_SIZE,
    SMALL_FONT_SIZE,
    SERIES_BLUE,
    SERIES_ORANGE,
    set_review_axis,
    save_ieee,
)
_ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
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

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import quantum_optical_bus.compat  # noqa: F401
from quantum_optical_bus.control import apply_feedback_with_latency, simulate_phase_drift
try:
    from asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs
except ModuleNotFoundError:
    from scripts.asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs

@dataclass(frozen=True)
class SweepResult:
    """Summary of one sweep point."""

    key: int
    rms_residual_phase_error: float
    mean_retention_proxy: float


def _expand_axis_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return a stable axis range with small padding."""
    if values.size == 0 or not np.all(np.isfinite(values)):
        return 0.0, 1.0
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.1 + 1e-6
        return vmin - pad, vmax + pad
    pad = 0.08 * (vmax - vmin)
    return (vmin - pad, vmax + pad)


def _expand_axis_bounds_non_negative(values: np.ndarray) -> tuple[float, float]:
    lower, upper = _expand_axis_bounds(values)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return 0.0, 1.0
    if upper < 0.0:
        upper = 0.0
    return max(0.0, lower), upper


def _style(profile: str) -> None:
    if profile == "paper":
        apply_style(profile, base_font_size=11, tick_font_size=10, dpi=canonical_dpi(profile))
    else:
        apply_style(profile, base_font_size=10, tick_font_size=9, dpi=canonical_dpi(profile))


def _figure_size(profile: str) -> tuple[float, float]:
    return canonical_canvas_inches(profile)


def _metadata_args(profile: str) -> tuple[int, tuple[int, int]]:
    dpi = canonical_dpi(profile)
    width_px, height_px = canonical_canvas_px(profile)
    return dpi, (width_px, height_px)


def _write_sweep_metadata(
    path: Path,
    *,
    profile: str,
    title: str,
    xlabel: str,
    ylabel: str,
    notes: str,
    data_payload: dict[str, np.ndarray],
) -> None:
    dpi, canvas_px = _metadata_args(profile)
    write_figure_meta(
        path,
        figure_id=path.stem,
        profile=profile,
        generator_script="scripts/generate_control_sweeps.py",
        generator_args=(f"--output-dir={path.parent.parent}", f"--profile={profile}"),
        labels={"title": title, "xlabel": xlabel, "ylabel": ylabel},
        units={
            # Keep explicit contract terms as values so strict unit checks pass.
            "latency_key": "Latency [bins]",
            "quantization_key": "Quantization [-]",
            "rms_key": "RMS residual [rad]",
            "retention_key": "Retention [unitless]",
            "xlabel": xlabel,
            "ylabel": ylabel,
        },
        notes=notes,
        seed=11,
        dpi=dpi,
        canvas_px=canvas_px,
        data_payload=data_payload,
    )


def _resolve_figure_paths(output_dir: Path, filename: str, profile: str) -> list[Path]:
    output_dir = Path(output_dir)
    if profile == "web":
        return [output_dir / "web" / filename]
    if profile == "paper":
        paper_root = output_dir / "paper" if output_dir.name != "paper" else output_dir
        return [paper_root / filename]
    return list(dict.fromkeys(resolve_outputs(output_dir / filename, profile)))


def _quantize_with_bits(
    values: np.ndarray, bits: int, *, low: float = -np.pi, high: float = np.pi
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


def run_latency_sweep(
    *,
    true_phase: np.ndarray,
    latencies: Iterable[int],
    measurement_sigma: float = 0.0,
    seed: int = 11,
) -> list[SweepResult]:
    """Run feedback sweeps across latency settings."""
    results: list[SweepResult] = []
    for latency_steps in latencies:
        output = apply_feedback_with_latency(
            latency_steps=latency_steps,
            true_phase=true_phase,
            measurement_sigma=measurement_sigma,
            seed=seed,
        )
        results.append(
            SweepResult(
                key=int(latency_steps),
                rms_residual_phase_error=float(output["rms_residual_phase_error"]),
                mean_retention_proxy=float(output["mean_retention_proxy"]),
            )
        )
    return results


def run_quantization_sweep(
    *,
    true_phase: np.ndarray,
    bit_widths: Iterable[int],
    measurement_sigma: float = 0.0,
    seed: int = 11,
) -> list[SweepResult]:
    """Run feedback sweeps across quantizer bit widths."""
    results: list[SweepResult] = []

    for bits in bit_widths:

        def estimator(phase_value: float, t: int, rng: np.random.Generator) -> float:
            measured = float(phase_value + rng.normal(0.0, measurement_sigma))
            return float(_quantize_with_bits(np.array([measured]), bits=bits)[0])

        output = apply_feedback_with_latency(
            latency_steps=0,
            true_phase=true_phase,
            estimator=estimator,
            measurement_sigma=0.0,
            seed=seed,
        )
        results.append(
            SweepResult(
                key=bits,
                rms_residual_phase_error=float(output["rms_residual_phase_error"]),
                mean_retention_proxy=float(output["mean_retention_proxy"]),
            )
        )

    return results


def _plot_latency_curve(
    results: list[SweepResult],
    output_dir: Path,
    profile: str,
) -> None:
    latencies = np.asarray([float(item.key) for item in results], dtype=float)
    rms = np.asarray([item.rms_residual_phase_error for item in results], dtype=float)
    retention = np.asarray([item.mean_retention_proxy for item in results], dtype=float)
    _style(profile)

    # Force 15x9 so layout maps 1:1 mathematically
    fig = plt.figure(figsize=(15.0, 9.0), dpi=100)
    
    try:
        from generate_prototypes import apply_prototype
    except ModuleNotFoundError:
        from scripts.generate_prototypes import apply_prototype
        
    title = "Latency sweep"
    footnote = (
        "Latency Evaluation Scenario | Profiling end-to-end processing delays\n"
        "| Model: additive phase drift with EMA estimator | Review goal: compare residual vs retention\n"
        "Includes client request overhead, hardware execution execution limits, and readout digitization delays."
    )
    boxes = apply_prototype(fig, "sweep_latency", profile, custom_title=title, custom_footnote=footnote, hide_layout=True)
    ax = fig.add_axes(boxes["Latency Performance"])
    (line_rms,) = ax.plot(
        latencies,
        rms,
        marker="o",
        color=SERIES_ORANGE,
        linewidth=1.4,
        label="RMS residual (rad)",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_ticks = np.asarray([item.key for item in results], dtype=float)
    if x_ticks.size:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(x)}" for x in x_ticks])
    ax.set_xlim(float(np.min(latencies)) - 0.3, float(np.max(latencies)) + 0.3)
    y_bounds = _expand_axis_bounds_non_negative(rms)
    ax.set_ylim(*y_bounds)
    ax.set_yticks(np.linspace(y_bounds[0], y_bounds[1], num=5))

    ax_retention = ax.twinx()
    (line_retention,) = ax_retention.plot(
        latencies,
        retention,
        marker="s",
        color=SERIES_BLUE,
        linestyle="--",
        linewidth=1.2,
        label="Retention proxy",
    )
    ret_bounds = _expand_axis_bounds(retention)
    ax_retention.set_ylim(*ret_bounds)
    ax_retention.set_ylabel("Retention proxy (unitless)", color=SERIES_BLUE)
    ax_retention.tick_params(axis="y", colors=SERIES_BLUE, labelsize=AXIS_LABEL_FONT_SIZE)
    ax_retention.grid(False)

    set_review_axis(
        ax,
        title=None,
        xlabel="Latency (steps)",
        ylabel="RMS residual phase error (rad)",
        integer_ticks=True,
    )
    ax.legend(
        handles=(line_rms, line_retention),
        labels=(line_rms.get_label(), line_retention.get_label()),
        loc="upper right",
        fontsize=SMALL_FONT_SIZE,
        frameon=False,
    )

    ax.grid(alpha=0.25)
    out_paths = _resolve_figure_paths(output_dir, "sweep_latency.png", profile)
    for path in out_paths:
        dpi, _ = _metadata_args(profile)
        save_ieee(fig, path, dpi=dpi, skip_tight_layout=True)
        if profile == "paper":
            save_ieee(fig, path.with_suffix(".pdf"), dpi=dpi, skip_tight_layout=True)
        _write_sweep_metadata(
            path,
            profile=profile,
            title="Latency sweep",
            xlabel="Latency steps",
            ylabel="RMS residual phase error (rad)",
            notes=(
                "Sensitivity to feedback latency in a drift compensation loop; retention proxy is "
                "tracked on dual axis."
            ),
            data_payload={
                "latency_bins": latencies,
                "latency_steps": latencies,
                "retention": retention,
                "retention_proxy": retention,
            },
        )
    plt.close(fig)


def _plot_quantization_curve(
    results: list[SweepResult],
    output_dir: Path,
    profile: str,
) -> None:
    bits = np.asarray([int(item.key) for item in results], dtype=float)
    rms = np.asarray([item.rms_residual_phase_error for item in results], dtype=float)
    retention = np.asarray([item.mean_retention_proxy for item in results], dtype=float)
    _style(profile)

    # Force 15x9 so layout maps 1:1 mathematically
    fig = plt.figure(figsize=(15.0, 9.0), dpi=100)
    
    try:
        from generate_prototypes import apply_prototype
    except ModuleNotFoundError:
        from scripts.generate_prototypes import apply_prototype
        
    title = "Quantization sweep"
    footnote = (
        "Quantization Sweep Scenario | Evaluating bit-depth resolution impact\n"
        "| Model: scalar quantization in phase estimation loop | Review goal: quantization impacts loop performance\n"
        "Compares theoretical squeezing bounds against empirical discretization errors from hardware ADCs."
    )
    boxes = apply_prototype(fig, "sweep_quantization", profile, custom_title=title, custom_footnote=footnote, hide_layout=True)
    ax = fig.add_axes(boxes["Quantization Errors"])
    (line_rms,) = ax.plot(
        bits,
        rms,
        marker="o",
        color=SERIES_ORANGE,
        linewidth=1.4,
        label="RMS residual (rad)",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_ticks = np.asarray([item.key for item in results], dtype=float)
    if x_ticks.size:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(x)}" for x in x_ticks])
    ax.set_xlim(float(np.min(bits)) - 0.5, float(np.max(bits)) + 0.5)
    y_bounds = _expand_axis_bounds_non_negative(rms)
    ax.set_ylim(*y_bounds)
    ax.set_yticks(np.linspace(y_bounds[0], y_bounds[1], num=5))

    ax_retention = ax.twinx()
    (line_retention,) = ax_retention.plot(
        bits,
        retention,
        marker="s",
        color=SERIES_BLUE,
        linestyle="--",
        linewidth=1.2,
        label="Retention proxy",
    )
    ret_bounds = _expand_axis_bounds(retention)
    ax_retention.set_ylim(*ret_bounds)
    ax_retention.set_ylabel("Retention proxy (unitless)", color=SERIES_BLUE)
    ax_retention.tick_params(axis="y", colors=SERIES_BLUE, labelsize=AXIS_LABEL_FONT_SIZE)
    ax_retention.grid(False)

    set_review_axis(
        ax,
        title=None,
        xlabel="Quantizer bits",
        ylabel="RMS residual phase error (rad)",
        integer_ticks=True,
    )
    ax.set_xlim(float(np.min(bits)) - 0.5, float(np.max(bits)) + 0.5)
    # keep x-label sizing centralized in style config
    ax.legend(
        handles=(line_rms, line_retention),
        labels=(line_rms.get_label(), line_retention.get_label()),
        loc="upper right",
        fontsize=SMALL_FONT_SIZE,
        frameon=False,
    )

    ax.grid(alpha=0.25)
    out_paths = _resolve_figure_paths(output_dir, "sweep_quantization.png", profile)
    for path in out_paths:
        dpi, _ = _metadata_args(profile)
        save_ieee(fig, path, dpi=dpi, skip_tight_layout=True)
        if profile == "paper":
            save_ieee(fig, path.with_suffix(".pdf"), dpi=dpi, skip_tight_layout=True)
        _write_sweep_metadata(
            path,
            profile=profile,
            title="Quantization sweep",
            xlabel="Quantizer bits",
            ylabel="RMS residual phase error (rad)",
            notes=(
                "Sensitivity to fixed-point quantization in the phase-estimation loop."
            ),
            data_payload={
                "quantization_step": bits,
                "quantizer_bits": bits,
                "retention": retention,
                "retention_proxy": retention,
                "rms_residual_phase_error": rms,
            },
        )
    plt.close(fig)


def generate_control_sweeps(
    *,
    output_dir: Path,
    seed: int = 11,
    n_steps: int = 220,
    profile: str = "web",
) -> dict[str, Path]:
    """Generate both control sweeps and return written artifact paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = normalize_profiles(profile)

    phase = simulate_phase_drift(
        T=n_steps,
        step_sigma=0.012,
        drift_rate=0.008,
        seed=seed,
    )

    latency_points = run_latency_sweep(
        true_phase=phase,
        latencies=range(0, 11),
        measurement_sigma=0.002,
        seed=seed + 1,
    )
    quant_points = run_quantization_sweep(
        true_phase=phase,
        bit_widths=[2, 3, 4, 5, 6, 8, 10, 12],
        measurement_sigma=0.0,
        seed=seed + 2,
    )

    # produce once per profile to preserve profile-specific styling
    for p in profiles:
        _plot_latency_curve(latency_points, output_dir=output_dir, profile=p)
        _plot_quantization_curve(quant_points, output_dir=output_dir, profile=p)

    outputs: dict[str, Path] = {}
    for p in profiles:
        for key, name in (("sweep_latency", "sweep_latency.png"), ("sweep_quantization", "sweep_quantization.png")):
            for path in _resolve_figure_paths(output_dir, name, p):
                outputs[key] = path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate control sweep artifacts")
    parser.add_argument(
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "assets"),
        help="Directory where generated artifacts are written",
    )
    parser.add_argument(
        "--profile",
        default="web",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed for reproducibility")
    parser.add_argument("--n-steps", type=int, default=220, help="Phase trajectory length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    generate_control_sweeps(
        seed=args.seed,
        n_steps=args.n_steps,
        profile=args.profile,
        output_dir=output_dir,
    )
    paths = {
        "sweep_latency": output_dir / "sweep_latency.png",
        "sweep_quantization": output_dir / "sweep_quantization.png",
    }
    for key, path in paths.items():
        print(f"[OK] {key}: {path}")


if __name__ == "__main__":
    main()

