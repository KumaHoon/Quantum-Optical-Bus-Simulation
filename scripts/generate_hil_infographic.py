"""Generate a Hardware-in-the-loop (HIL) expansion infographic."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scripts.viz_style_ieee import (
    BG_COLOR,
    SERIES_BLUE,
    SERIES_ORANGE,
    SERIES_PURPLE,
    SERIES_TEAL,
    apply_ieee_style,
    save_ieee,
    ieee_figsize,
)

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_PNG = ROOT_DIR / "docs" / "figures" / "hil_expansion.png"
OUTPUT_PDF = ROOT_DIR / "docs" / "figures" / "hil_expansion.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-png",
        type=pathlib.Path,
        default=OUTPUT_PNG,
        help="Output infographic PNG path (default: docs/figures/hil_expansion.png).",
    )
    parser.add_argument(
        "--output-pdf",
        type=pathlib.Path,
        default=OUTPUT_PDF,
        help="Optional output PDF path (default: docs/figures/hil_expansion.pdf).",
    )
    return parser.parse_args()


def _lane_title(ax: plt.Axes, y: float, title: str, subtitle: str) -> None:
    ax.text(0.0, y + 0.04, title, fontsize=11, fontweight="bold", color="#c9d1d9")
    ax.text(0.0, y - 0.01, subtitle, fontsize=8.5, color="#8b949e")


def _node(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    tag: str,
    *, 
    color: str,
) -> None:
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05",
        linewidth=1.15,
        facecolor="#161b22",
        edgecolor=color,
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + 0.74 * h,
        title,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#c9d1d9",
        fontweight="bold",
    )
    ax.text(
        x + w / 2,
        y + 0.42 * h,
        subtitle,
        ha="center",
        va="center",
        fontsize=7.5,
        color="#8b949e",
    )
    ax.text(
        x + w / 2,
        y + 0.12 * h,
        tag,
        ha="center",
        va="center",
        fontsize=6.9,
        color="#58a6ff",
        fontstyle="italic",
    )


def _arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.4,
            color=color,
            shrinkA=1,
            shrinkB=1,
        ),
    )


def generate() -> plt.Figure:
    apply_ieee_style(base_font_size=10, tick_font_size=9)
    fig, ax = plt.subplots(figsize=ieee_figsize(width_in=7.16, aspect=0.60))
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    # Optical lane
    y_opt = 0.71
    _lane_title(ax, y_opt + 0.08, "Optical Lane", "Classical laser/OPO + photodetection path")
    nx = 0.12
    nodes_opt = [
        ("Laser", "pump source\nsetpoint", "calibration_app.py", SERIES_BLUE),
        ("OPA", "squeezing /\nloop gain", "interface.py", SERIES_TEAL),
        ("Loop", "propagation +\nhomodyne", "hardware.py", SERIES_ORANGE),
        ("Readout", "sampled quadratures", "calibration_app.py", SERIES_PURPLE),
    ]
    w = 0.155
    h = 0.08
    x = nx
    gap = 0.04
    for idx, (title, sub, tag, color) in enumerate(nodes_opt):
        _node(ax, x, y_opt, w, h, title, sub, tag, color=color)
        if idx < len(nodes_opt) - 1:
            _arrow(ax, x + w, y_opt + h / 2, x + w + gap, y_opt + h / 2, color=color)
            x += w + gap
        else:
            x += w

    # Control lane
    y_ctrl = 0.41
    _lane_title(ax, y_ctrl + 0.08, "Control Lane", "Adaptive DSP / feedback path")
    nx = 0.08
    nodes_ctrl = [
        ("ADC", "sample ADC", "hardware.py / sf", SERIES_BLUE),
        ("FPGA", "estimator", "control.py", SERIES_TEAL),
        ("DAC", "actuation", "control.py", SERIES_ORANGE),
        ("EOM", "phase control", "control.py", SERIES_PURPLE),
    ]
    x = nx
    for idx, (title, sub, tag, color) in enumerate(nodes_ctrl):
        _node(ax, x, y_ctrl, w, h, title, sub, tag, color=color)
        if idx < len(nodes_ctrl) - 1:
            _arrow(ax, x + w, y_ctrl + h / 2, x + w + gap, y_ctrl + h / 2, color=color)
            x += w + gap
        else:
            x += w

    # World model lane
    y_wm = 0.11
    _lane_title(ax, y_wm + 0.08, "World-Model Lane", "Model extraction + deployment")
    nx = 0.05
    nodes_wm = [
        ("Logs", "state snapshots", "streamlit", SERIES_BLUE),
        ("estimation", "fit_eta_and_loss", "estimation.py", SERIES_TEAL),
        ("Model update", "coefficients", "interface.py", SERIES_ORANGE),
        ("Deploy", "control coefficients", "hdl/\n(sim pending)", SERIES_PURPLE),
    ]
    x = nx
    for idx, (title, sub, tag, color) in enumerate(nodes_wm):
        _node(ax, x, y_wm, w, h, title, sub, tag, color=color)
        if idx < len(nodes_wm) - 1:
            _arrow(ax, x + w, y_wm + h / 2, x + w + gap, y_wm + h / 2, color=color)
            x += w + gap
        else:
            x += w

    # Inter-lane coupling arrows
    _arrow(ax, 0.33, 0.7, 0.33, 0.59, SERIES_ORANGE)
    _arrow(ax, 0.50, 0.7, 0.50, 0.59, SERIES_ORANGE)
    _arrow(ax, 0.67, 0.7, 0.67, 0.59, SERIES_ORANGE)
    _arrow(ax, 0.33, 0.4, 0.33, 0.29, SERIES_ORANGE)
    _arrow(ax, 0.50, 0.4, 0.50, 0.29, SERIES_ORANGE)
    _arrow(ax, 0.67, 0.4, 0.67, 0.29, SERIES_ORANGE)
    _arrow(ax, 0.67, 0.83, 0.67, 0.92, SERIES_BLUE)

    ax.text(
        0.5,
        0.95,
        "Future Hardware-in-the-Loop Expansion",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#c9d1d9",
    )
    fig.tight_layout()
    return fig


def save_outputs(fig: plt.Figure, out_png: pathlib.Path, out_pdf: pathlib.Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_ieee(fig, out_png, dpi=300)
    try:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, dpi=300)
        print(f"[OK] Saved HIL infographic PDF: {out_pdf}")
    except Exception:
        # PDF rendering can fail in minimal environments; keep PNG-only fallback.
        print("[INFO] Skipping PDF export.")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fig = generate()
    save_outputs(fig, args.output_png, args.output_pdf)
    print(f"[OK] Saved HIL infographic PNG: {args.output_png}")


if __name__ == "__main__":
    main()
