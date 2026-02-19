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

from quantum_optical_bus.viz_style_ieee import (
    FIGURE_WIDTH_2COL_IN,
    apply_ieee_style,
    save_ieee,
    ieee_figsize,
    style_axis,
)

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import quantum_optical_bus.compat  # noqa: F401
from quantum_optical_bus.control import apply_feedback_with_latency, simulate_phase_drift


@dataclass(frozen=True)
class SweepResult:
    """Summary of one sweep point."""

    key: int
    rms_residual_phase_error: float
    mean_retention_proxy: float


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


def _plot_latency_curve(results: list[SweepResult], output_path: Path) -> None:
    latencies = [float(item.key) for item in results]
    rms = [item.rms_residual_phase_error for item in results]
    apply_ieee_style(base_font_size=10, tick_font_size=9)

    fig, ax = plt.subplots(figsize=ieee_figsize(width_in=FIGURE_WIDTH_2COL_IN, aspect=0.56))
    ax.plot(latencies, rms, marker="o", linewidth=2, label="RMS residual")
    style_axis(
        ax,
        title="Control Latency Sweep",
        xlabel="Latency (bins / steps)",
        ylabel="RMS residual phase error (rad)",
    )
    fig.tight_layout()
    save_ieee(fig, output_path.with_name("sweep_latency.png"), dpi=300)
    plt.close(fig)


def _plot_quantization_curve(results: list[SweepResult], output_path: Path) -> None:
    bits = [int(item.key) for item in results]
    rms = [item.rms_residual_phase_error for item in results]
    apply_ieee_style(base_font_size=10, tick_font_size=9)

    fig, ax = plt.subplots(figsize=ieee_figsize(width_in=FIGURE_WIDTH_2COL_IN, aspect=0.56))
    ax.plot(bits, rms, marker="o", linewidth=2)
    style_axis(
        ax,
        title="Quantization Sweep",
        xlabel="Quantizer bits (unitless)",
        ylabel="RMS residual phase error (rad)",
    )
    fig.tight_layout()
    save_ieee(fig, output_path.with_name("sweep_quantization.png"), dpi=300)
    plt.close(fig)


def generate_control_sweeps(
    *,
    output_dir: Path,
    seed: int = 11,
    n_steps: int = 220,
) -> dict[str, Path]:
    """Generate both control sweeps and return written artifact paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

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

    _plot_latency_curve(latency_points, output_dir / "sweep_latency.png")
    _plot_quantization_curve(quant_points, output_dir / "sweep_quantization.png")

    return {
        "sweep_latency": output_dir / "sweep_latency.png",
        "sweep_quantization": output_dir / "sweep_quantization.png",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate control sweep artifacts")
    parser.add_argument(
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "assets"),
        help="Directory where generated artifacts are written",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed for reproducibility")
    parser.add_argument("--n-steps", type=int, default=220, help="Phase trajectory length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paths = generate_control_sweeps(seed=args.seed, n_steps=args.n_steps, output_dir=output_dir)
    for key, path in paths.items():
        print(f"[OK] {key}: {path}")


if __name__ == "__main__":
    main()
