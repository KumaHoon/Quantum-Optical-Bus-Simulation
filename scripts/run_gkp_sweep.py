"""Generate a reproducible GKP toy proxy sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

from quantum_optical_bus.viz_style_ieee import (
    FIGURE_WIDTH_2COL_IN,
    ieee_figsize,
    apply_review_layout,
    set_review_axis,
    set_tab_title,
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
    from figstyle import apply_style, canonical_canvas_px, canonical_dpi, write_figure_meta
except ModuleNotFoundError:
    from scripts.figstyle import apply_style, canonical_canvas_px, canonical_dpi, write_figure_meta

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

SQRT_PI_HALF = np.sqrt(np.pi) / 2.0


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
    ratio = SQRT_PI_HALF / (np.sqrt(2.0) * sigma_total)
    return erfc(ratio)


def _apply_profile(profile: str) -> None:
    apply_style(
        profile,
        base_font_size=11 if profile == "paper" else 10,
        tick_font_size=10 if profile == "paper" else 9,
        dpi=canonical_dpi(profile),
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


def run_gkp_sweep(
    *,
    output_path: Path,
    squeezing_points: int = 14,
    noise_points: int = 120,
    profile: str = "web",
) -> Path:
    """Run a GKP toy sweep and save the plot under a stable filename."""
    if squeezing_points < 2 or noise_points < 3:
        raise ValueError("Need enough points for a meaningful sweep")

    profiles = normalize_profiles(profile)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    squeezing = np.linspace(0.2, 1.3, squeezing_points)
    noise = np.linspace(0.01, 0.9, noise_points)
    noise_grid, squeeze_grid = np.meshgrid(noise, squeezing)

    logical_error = logical_error_proxy(squeeze_grid, noise_grid)
    baseline_noise = logical_error[0]
    strong_squeeze = logical_error[-1]

    name = output_path.name
    base = output_path.parent
    for p in profiles:
        _apply_profile(p)
        fig, axes = plt.subplots(
            1, 2, figsize=ieee_figsize(width_in=FIGURE_WIDTH_2COL_IN, aspect=0.56)
        )

        im = axes[0].pcolormesh(noise, squeezing, logical_error, shading="auto")
        axes[0].set_title("Noise and squeezing proxy", fontsize=8)
        axes[0].set_xlabel("Shift-noise std")
        axes[0].set_ylabel("Squeezing parameter r")
        axes[0].set_aspect("auto")
        cb = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cb.set_label("Logical error proxy", fontsize=7)
        cb.ax.tick_params(labelsize=6, pad=1)

        axes[1].plot(noise, baseline_noise, lw=1.4, label=f"r={squeezing[0]:.2f}")
        axes[1].plot(noise, strong_squeeze, lw=1.4, label=f"r={squeezing[-1]:.2f}")
        set_review_axis(
            axes[0],
            title=axes[0].get_title(),
            xlabel=axes[0].get_xlabel(),
            ylabel=axes[0].get_ylabel(),
            integer_ticks=False,
        )
        axes[1].set_title("Proxy slices", fontsize=8)
        set_review_axis(
            axes[1],
            title=axes[1].get_title() if hasattr(axes[1], "get_title") else "Noise slice",
            xlabel="Shift-noise std",
            ylabel="Logical error proxy",
            integer_ticks=False,
        )
        axes[1].set_xlabel("Shift-noise std")
        axes[1].set_ylabel("Logical error proxy")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="upper right", fontsize=7, frameon=False)

        set_tab_title(fig, "GKP Toy Proxy (Supporting Intuition)", mode=p)
        fig.text(
            0.07,
            0.02,
            "Toy model: no full GKP encoding/decoding stack; used as auxiliary evidence only.",
            ha="left",
            va="bottom",
            fontsize=7,
            color="#8b949e",
        )
        apply_review_layout(
            fig,
            mode=p,
            left=0.08,
            right=0.98,
            bottom=0.08,
            top=0.90,
            wspace=0.20,
            hspace=0.20,
        )

        targets = _resolve_output_targets(output_path, p)
        for target, target_profile in targets:
            target_dpi = canonical_dpi(target_profile)
            if target_profile == "paper":
                save_ieee(fig, target.with_suffix(".pdf"), dpi=target_dpi, skip_tight_layout=True)
                save_ieee(fig, target, dpi=target_dpi, skip_tight_layout=True)
            else:
                save_ieee(fig, target, dpi=target_dpi, skip_tight_layout=True)
            write_figure_meta(
                target,
                figure_id=target.stem,
                profile=target_profile,
                generator_script="scripts/run_gkp_sweep.py",
                generator_args=(
                    f"--output-dir={output_path.parent}",
                    f"--profile={target_profile}",
                ),
                labels={
                    "title": "GKP toy proxy",
                    "xlabel": "Shift-noise std",
                    "ylabel": "Logical error proxy",
                },
                units={"x": "dimensionless", "y": "probability"},
                notes="Auxiliary GKP intuition figure (supplemental).",
                seed=17,
                dpi=canonical_dpi(target_profile),
                canvas_px=canonical_canvas_px(target_profile),
            )
        plt.close(fig)

    return base / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a minimal GKP toy proxy sweep.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory where gkp_proxy.png is written.",
    )
    parser.add_argument(
        "--output",
        default="gkp_proxy.png",
        help="Output PNG path or filename.",
    )
    parser.add_argument("--squeezing-points", type=int, default=14)
    parser.add_argument("--noise-points", type=int, default=120)
    parser.add_argument(
        "--profile",
        default="web",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_arg = Path(args.output)
    output_path = (
        args.output_dir / output_arg.name if output_arg.parent == Path(".") else output_arg
    )

    path = run_gkp_sweep(
        output_path=output_path,
        squeezing_points=args.squeezing_points,
        noise_points=args.noise_points,
        profile=args.profile,
    )
    print(f"[OK] wrote {path}")


if __name__ == "__main__":
    main()
