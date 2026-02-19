"""Generate a reproducible HIL infographic."""

from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
STYLE_PATH = ROOT_DIR / "src" / "quantum_optical_bus" / "viz_style_ieee.py"


def _load_style_module():
    spec = importlib.util.spec_from_file_location("quantum_optical_bus.viz_style_ieee", STYLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load IEEE style module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STYLE = _load_style_module()
AXIS_COLOR = _STYLE.AXIS_COLOR
BG_COLOR = _STYLE.BG_COLOR
PANEL_COLOR = _STYLE.PANEL_COLOR
SERIES_BLUE = _STYLE.SERIES_BLUE
SERIES_ORANGE = _STYLE.SERIES_ORANGE
SERIES_PURPLE = _STYLE.SERIES_PURPLE
SERIES_TEAL = _STYLE.SERIES_TEAL
ieee_figsize = _STYLE.ieee_figsize
save_ieee = _STYLE.save_ieee
set_ieee_rcparams = _STYLE.set_ieee_rcparams

DEFAULT_PNG = ROOT_DIR / "docs" / "figures" / "hil_expansion.png"
DEFAULT_PDF = ROOT_DIR / "docs" / "figures" / "hil_expansion.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_PNG,
        help="Output infographic PNG path (default: docs/figures/hil_expansion.png)",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Optional output PDF path (default: docs/figures/hil_expansion.pdf)",
    )
    return parser.parse_args()


def _node(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    tag: str,
    *,
    color: str,
) -> None:
    box = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04",
        linewidth=1.1,
        edgecolor=color,
        facecolor=PANEL_COLOR,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + 0.74 * height,
        title,
        ha="center",
        va="top",
        fontsize=9.5,
        color="white",
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + 0.44 * height,
        subtitle,
        ha="center",
        va="top",
        fontsize=7.8,
        color=AXIS_COLOR,
    )
    ax.text(
        x + width / 2,
        y + 0.08 * height,
        tag,
        ha="center",
        va="top",
        fontsize=7,
        color=AXIS_COLOR,
        fontstyle="italic",
    )


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
    )


def _draw_lane_title(ax: plt.Axes, x: float, y: float, title: str) -> None:
    ax.text(
        x,
        y,
        title,
        fontsize=11,
        fontweight="bold",
        color="white",
        va="bottom",
    )


def generate() -> plt.Figure:
    set_ieee_rcparams(base_font_size=10, tick_font_size=9)
    fig, ax = plt.subplots(figsize=ieee_figsize(width_in=7.16, aspect=0.62))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor(BG_COLOR)

    block_w = 0.17
    block_h = 0.085
    gap_x = 0.035

    # Optical lane: Laser/OPA/Loop/Homodyne
    y_optical = 0.73
    _draw_lane_title(ax, 0.02, y_optical + block_h + 0.04, "Optical lane")
    x = 0.09
    optical_blocks = [
        ("Laser", "pump source", "calibration_app.py", SERIES_BLUE),
        ("OPA", "pump transfer", "interface.py", SERIES_TEAL),
        ("Loop", "propagation", "hardware.py", SERIES_ORANGE),
        ("Homodyne", "quadrature readout", "calibration_app.py", SERIES_PURPLE),
    ]
    for idx, (title, subtitle, tag, color) in enumerate(optical_blocks):
        _node(ax, x, y_optical, block_w, block_h, title, subtitle, tag, color=color)
        if idx < len(optical_blocks) - 1:
            _arrow(
                ax,
                x + block_w,
                y_optical + block_h / 2,
                x + block_w + gap_x,
                y_optical + block_h / 2,
                color=color,
            )
        x += block_w + gap_x

    # Control lane: ADC/FPGA/DAC/EOM driver
    y_control = 0.43
    _draw_lane_title(ax, 0.02, y_control + block_h + 0.04, "Control lane")
    x = 0.09
    control_blocks = [
        ("ADC", "sampled measurements", "hardware.py", SERIES_BLUE),
        ("FPGA", "estimate residuals", "control.py", SERIES_TEAL),
        ("DAC", "actuation output", "control.py", SERIES_ORANGE),
        ("EOM driver", "phase correction", "control.py", SERIES_PURPLE),
    ]
    for idx, (title, subtitle, tag, color) in enumerate(control_blocks):
        _node(ax, x, y_control, block_w, block_h, title, subtitle, tag, color=color)
        if idx < len(control_blocks) - 1:
            _arrow(
                ax,
                x + block_w,
                y_control + block_h / 2,
                x + block_w + gap_x,
                y_control + block_h / 2,
                color=color,
            )
        x += block_w + gap_x

    # World-model lane: logs -> estimation -> model update -> coefficient export -> deploy
    y_model = 0.13
    _draw_lane_title(ax, 0.02, y_model + block_h + 0.04, "World-model lane")
    x = 0.09
    model_blocks = [
        ("Logs", "data collection", "calibration_app.py", SERIES_BLUE),
        ("Estimation", "fit_eta_and_loss", "estimation.py", SERIES_TEAL),
        ("Model update", "update coefficients", "interface.py", SERIES_ORANGE),
        ("Coefficient export", "new gains", "control.py", SERIES_BLUE),
        ("Deploy", "hdl/ path", "hdl/", SERIES_TEAL),
    ]
    for idx, (title, subtitle, tag, color) in enumerate(model_blocks):
        _node(ax, x, y_model, block_w, block_h, title, subtitle, tag, color=color)
        if idx < len(model_blocks) - 1:
            _arrow(
                ax,
                x + block_w,
                y_model + block_h / 2,
                x + block_w + gap_x,
                y_model + block_h / 2,
                color=color,
            )
        x += block_w + gap_x

    # Cross-lane links
    _arrow(ax, 0.29, 0.73 + block_h / 2, 0.29, 0.43 + block_h / 2, SERIES_ORANGE)
    _arrow(ax, 0.54, 0.73 + block_h / 2, 0.54, 0.43 + block_h / 2, SERIES_ORANGE)
    _arrow(ax, 0.79, 0.73 + block_h / 2, 0.79, 0.43 + block_h / 2, SERIES_ORANGE)
    _arrow(ax, 0.45, 0.43 + block_h / 2, 0.45, 0.13 + block_h / 2, SERIES_TEAL)
    _arrow(ax, 0.70, 0.43 + block_h / 2, 0.70, 0.13 + block_h / 2, SERIES_TEAL)
    _arrow(ax, 0.35, 0.13 + block_h + 0.01, 0.35, 0.73 + 0.08, SERIES_PURPLE)

    ax.text(
        0.5,
        0.95,
        "Future Hardware-in-the-Loop Expansion",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="white",
    )
    fig.tight_layout()
    return fig


def save_outputs(fig: plt.Figure, output_png: Path, output_pdf: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    save_ieee(fig, output_png, dpi=300)
    try:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved HIL infographic PDF: {output_pdf}")
    except Exception:
        print("[INFO] PDF export skipped.")


def main() -> None:
    args = parse_args()
    fig = generate()
    save_outputs(fig, args.output_png, args.output_pdf)
    print(f"[OK] Saved HIL infographic PNG: {args.output_png}")


if __name__ == "__main__":
    main()
