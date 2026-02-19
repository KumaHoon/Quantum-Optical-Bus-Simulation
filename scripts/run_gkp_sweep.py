"""Generate a reproducible GKP toy proxy sweep."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

from quantum_optical_bus.viz_style_ieee import (
    FIGURE_WIDTH_2COL_IN,
    apply_ieee_style,
    ieee_figsize,
    save_ieee,
    style_axis,
)


SQRT_PI_HALF = math.sqrt(math.pi) / 2.0


def logical_error_proxy(squeezing: np.ndarray, noise_std: np.ndarray) -> np.ndarray:
    """Toy logical error proxy from finite squeezing and additive shift noise.

    Parameters
    ----------
    squeezing
        Squeezing values (r), dimensionless.
    noise_std
        Additive shift noise standard deviation.
    """
    r = np.asarray(squeezing, dtype=float)
    sigma_shift = np.asarray(noise_std, dtype=float)
    if np.any(sigma_shift < 0):
        raise ValueError("noise_std must be non-negative")

    intrinsic_std = np.sqrt(0.5) * np.exp(-r)
    sigma_total = np.sqrt(intrinsic_std**2 + sigma_shift**2)
    ratio = SQRT_PI_HALF / (math.sqrt(2.0) * sigma_total)
    return erfc(ratio)


def run_gkp_sweep(
    *,
    output_path: Path,
    squeezing_points: int = 14,
    noise_points: int = 120,
) -> Path:
    """Run a GKP toy sweep and save the plot under a stable filename."""
    if squeezing_points < 2 or noise_points < 3:
        raise ValueError("Need enough points for a meaningful sweep")

    squeezing = np.linspace(0.2, 1.3, squeezing_points)  # q gate parameter r
    noise = np.linspace(0.01, 0.9, noise_points)  # shift-noise proxy
    noise_grid, squeeze_grid = np.meshgrid(noise, squeezing)

    logical_error = logical_error_proxy(squeeze_grid, noise_grid)
    baseline_noise = logical_error[0]
    strong_squeeze = logical_error[-1]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    apply_ieee_style(base_font_size=10, tick_font_size=9)
    fig, axes = plt.subplots(1, 2, figsize=ieee_figsize(width_in=FIGURE_WIDTH_2COL_IN, aspect=0.56))

    im = axes[0].pcolormesh(noise, squeezing, logical_error, shading="auto")
    axes[0].set_title("Noise and squeezing to logical error proxy")
    axes[0].set_xlabel("Shift-noise std (unitless)")
    axes[0].set_ylabel("Squeezing parameter r (unitless)")
    axes[0].set_aspect("auto")
    cb = plt.colorbar(im, ax=axes[0])
    cb.set_label("Logical error proxy = P(|shift| > sqrt(pi)/2) (unitless)")

    axes[1].plot(noise, baseline_noise, lw=1.8, label=f"r={squeezing[0]:.2f}")
    axes[1].plot(noise, strong_squeeze, lw=1.8, label=f"r={squeezing[-1]:.2f}")
    style_axis(
        axes[0],
        title=axes[0].get_title(),
        xlabel=axes[0].get_xlabel(),
        ylabel=axes[0].get_ylabel(),
    )
    axes[1].set_title("Noise slice")
    axes[1].set_xlabel("Shift-noise std (unitless)")
    axes[1].set_ylabel("Logical error proxy (unitless)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("GKP Toy: finite squeezing + shift noise proxy", fontsize=12)
    fig.tight_layout()

    save_ieee(fig, output_path, dpi=300)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal GKP toy proxy sweep.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "gkp_proxy.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--squeezing-points", type=int, default=14)
    parser.add_argument("--noise-points", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = run_gkp_sweep(
        output_path=Path(args.output),
        squeezing_points=args.squeezing_points,
        noise_points=args.noise_points,
    )
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
