"""Generate prototype skeleton (wireframe) diagrams of the existing assets."""

import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure the parent directory is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Constants for design
BG_COLOR = "#12161c"
FRAME_COLOR = "#4a5568"
TEXT_COLOR = "#a0aec0"
HIGHLIGHT_COLOR = "#00b5d8"
PAPER_TEXT_COLOR = "#0f172a"
PAPER_TITLE_COLOR = "#1f2937"

OUTPUT_DIR = ROOT_DIR / "assets" / "abstract_prototype"

# Layout definitions (relative x, y, width, height)
LAYOUTS = {
    "dashboard_vacuum": [
        (35, 30, 1430, 840, "Main Dashboard Wrapper", "wrapper"),
        (55, 790, 1390, 40, "Title", "title_box"),
        (55, 500, 1390, 253, "Top Section", "inner_wrapper"),
        (55, 173, 1390, 253, "Bottom Section", "inner_wrapper"),
        (180, 500, 253, 253, "Waveguide", "box"),
        (640, 500, 680, 253, "Calibration Curve", "box"),
        (180, 173, 253, 253, "Wigner", "box"),
        (640, 173, 680, 253, "Photon Number", "box"),
        (
            55,
            50,
            1390,
            80,
            "Scenario 1: Vacuum baseline (0 mW) | Pump power: 0.0 mW | Intrinsic squeezing: ≈ 0 dB | Var(x)=Var(p)=0.5 (vac=0.5)\nVacuum reference scenario with pump power at 0 mW.",
            "footnote",
        ),
    ],
    "dashboard_calibration": [
        (35, 30, 1430, 840, "Main Dashboard Wrapper", "wrapper"),
        (55, 790, 1390, 40, "Title", "title_box"),
        (55, 506, 1390, 293, "Top Section", "inner_wrapper"),
        (55, 179, 1390, 253, "Bottom Section", "inner_wrapper"),
        (180, 506, 253, 253, "Waveguide", "box"),
        (640, 506, 680, 253, "Calibration Curve", "box"),
        (180, 173, 253, 253, "Wigner", "box"),
        (640, 173, 680, 253, "Photon Number", "box"),
        (
            55,
            50,
            1390,
            80,
            "Scenario 2: Squeezed state (200 mW) | Pump: 200 mW | Intrinsic squeezing: 8.45 dB | Photon cutoff: n=14 | P(n) normalized.\nSqueezed-state calibration scenario using an intrinsic operating point and photon-number diagnostics.",
            "footnote",
        ),
    ],
    "dashboard_decoherence": [
        (35, 30, 1430, 840, "Main Dashboard Wrapper", "wrapper"),
        (55, 790, 1390, 40, "Title", "title_box"),
        (55, 500, 1390, 253, "Top Section", "inner_wrapper"),
        (55, 173, 1390, 253, "Bottom Section", "inner_wrapper"),
        (295, 500, 910, 240, "Variance Sweep", "box"),
        (360, 173, 253, 253, "Wigner Pure", "box"),
        (872, 173, 253, 253, "Wigner Loss", "box"),
        (
            55,
            50,
            1390,
            80,
            "Scenario 3: Decoherence + loss | Pump power: 200.0 mW | Squeezing: r=0.97 | Transmissivity: 0.89 | Pre-loss: 8.45 dB | Post-loss: 5.09 dB\nLoss moves squeezing toward vacuum limit (Var = 0.5). Intrinsic magnitude is always >= observed after loss.",
            "footnote",
        ),
    ],
    "sweep_latency": [
        (35, 30, 1430, 840, "Wrapper", "wrapper"),
        (55, 790, 1390, 40, "Title", "title_box"),
        (135, 250, 1230, 500, "Latency Performance", "box"),
        (
            55,
            50,
            1390,
            80,
            "Latency Evaluation Scenario | Profiling end-to-end processing delays\nIncludes client request overhead, hardware execution execution limits, and readout digitization delays.",
            "footnote",
        ),
    ],
    "sweep_quantization": [
        (35, 30, 1430, 840, "Wrapper", "wrapper"),
        (55, 790, 1390, 40, "Title", "title_box"),
        (135, 250, 1230, 500, "Quantization Errors", "box"),
        (
            55,
            50,
            1390,
            80,
            "Quantization Sweep Scenario | Evaluating bit-depth resolution impact\nCompares theoretical squeezing bounds against empirical discretization errors from hardware ADCs.",
            "footnote",
        ),
    ],
    "calibration_demo": [
        (35, 30, 1430, 840, "Main Dashboard Wrapper", "wrapper"),
        (55, 790, 1390, 40, "Title", "title_box"),
        (55, 497, 1390, 293, "Top Section", "inner_wrapper"),
        (55, 170, 1390, 180, "Bottom Section", "inner_wrapper"),
        (200, 497, 700, 253, "Dashboard Metrics", "box"),
        (1000, 450, 300, 300, "Wigner Function", "box"),
        (200, 170, 1100, 180, "Calibration Sweep Phase", "box"),
        (
            55,
            20,
            1390,
            80,
            "Calibration Demo Scenario | Phase sweeps over operating points\nDemonstrates live feedback loops tracking maximum quadrature squeezing margins.",
            "footnote",
        ),
    ],
}


def apply_prototype(
    fig: plt.Figure,
    layout_name: str,
    profile: str = "web",
    *,
    custom_title: str | None = None,
    custom_footnote: str | None = None,
    hide_layout: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    """Draw abstract prototype borders on the figure and return normalized coordinates for data panels."""
    try:
        boxes = LAYOUTS[layout_name]
    except KeyError:
        raise ValueError(f"Unknown prototype layout: {layout_name}")

    CANVAS_W = 1500.0
    CANVAS_H = 900.0

    bg_ax = fig.add_axes([0, 0, 1, 1])
    bg_ax.axis("off")

    axes_coords = {}

    for x, y, w, h, label, btype in boxes:
        norm_x, norm_y = x / CANVAS_W, y / CANVAS_H
        norm_w, norm_h = w / CANVAS_W, h / CANVAS_H

        if btype == "box":
            axes_coords[label] = (norm_x, norm_y, norm_w, norm_h)
            continue

        if btype == "wrapper":
            if hide_layout:
                continue
            is_foot_wrapper = "foot" in label.lower()
            if profile == "paper" and is_foot_wrapper:
                rect = patches.Rectangle(
                    (norm_x, norm_y),
                    norm_w,
                    norm_h,
                    linewidth=2.5,
                    edgecolor="#718096",
                    facecolor="none",
                    alpha=0.6,
                    linestyle="dashdot",
                    transform=bg_ax.transAxes,
                )
                bg_ax.add_patch(rect)
        elif btype == "inner_wrapper":
            if hide_layout:
                continue
            # Wrapper labels intentionally hidden to keep the skeleton clean.
            continue
        elif btype in ("title", "title_box"):
            display_label = custom_title if custom_title is not None else label
            if not str(display_label).strip():
                continue
            if (
                btype == "title_box"
                and not hide_layout
                and profile == "web"
                and layout_name != "calibration_demo"
            ):
                rect = patches.Rectangle(
                    (norm_x, norm_y),
                    norm_w,
                    norm_h,
                    linewidth=1.5,
                    edgecolor="#a0aec0",
                    facecolor="#1a202c",
                    alpha=0.6,
                    linestyle="solid",
                    transform=bg_ax.transAxes,
                )
                bg_ax.add_patch(rect)
            bg_ax.text(
                norm_x + norm_w / 2,
                norm_y + norm_h / 2,
                display_label,
                ha="center",
                va="center",
                fontsize=11 if profile == "paper" else 28,
                color=PAPER_TITLE_COLOR if profile == "paper" else TEXT_COLOR,
                fontweight="bold",
                transform=bg_ax.transAxes,
            )
        elif btype == "footnote":
            display_label = custom_footnote if custom_footnote is not None else label
            footnote_font_size = 8 if profile == "paper" else 13
            if layout_name == "calibration_demo" and profile == "web":
                footnote_font_size = 12
            if profile == "web":
                rect = patches.Rectangle(
                    (norm_x, norm_y),
                    norm_w,
                    norm_h,
                    linewidth=1.5,
                    edgecolor="#a0aec0",
                    facecolor="#1a202c",
                    alpha=0.6,
                    linestyle="solid",
                    transform=bg_ax.transAxes,
                )
                bg_ax.add_patch(rect)
            bg_ax.text(
                norm_x + 15 / CANVAS_W,
                norm_y + norm_h / 2,
                display_label,
                ha="left",
                va="center",
                fontsize=footnote_font_size,
                color=PAPER_TEXT_COLOR if profile == "paper" else "#a0aec0",
                fontstyle="normal" if profile == "paper" else "italic",
                transform=bg_ax.transAxes,
            )

    return axes_coords


def draw_prototype(name: str, boxes: list[tuple[float, float, float, float, str, str]]) -> None:
    """Draw a prototype blueprint for a given asset using absolute pixel coordinates."""
    # 1500x900 at 100 DPI translates exactly to matching 1 pixel to 1 data unit
    fig, ax = plt.subplots(figsize=(15, 9), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 900)
    ax.axis("off")

    for x, y, w, h, label, btype in boxes:
        if btype == "wrapper":
            if "foot" in label.lower():
                rect = patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    linewidth=2.5,
                    edgecolor="#718096",
                    facecolor="none",
                    alpha=0.6,
                    linestyle="dashdot",
                )
                ax.add_patch(rect)
        elif btype == "inner_wrapper":
            # Wrapper labels intentionally hidden to keep the skeleton clean.
            pass
        elif btype == "title":
            ax.text(
                x + w / 2,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=36,
                color=TEXT_COLOR,
                fontweight="bold",
            )
        elif btype == "title_box":
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.5,
                edgecolor="#a0aec0",
                facecolor="#1a202c",
                alpha=0.6,
                linestyle="solid",
            )
            ax.add_patch(rect)
            ax.text(
                x + w / 2,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=36,
                color=TEXT_COLOR,
                fontweight="bold",
            )
        elif btype == "footnote":
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.5,
                edgecolor="#a0aec0",
                facecolor="#1a202c",
                alpha=0.6,
                linestyle="solid",
            )
            ax.add_patch(rect)
            ax.text(
                x + 15,
                y + h / 2,
                label,
                ha="left",
                va="center",
                fontsize=14,
                color="#a0aec0",
                fontstyle="italic",
            )
        else:  # "box"
            # Draw the rectangle
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor=FRAME_COLOR,
                facecolor="#1a202c",
                alpha=0.8,
                linestyle="--",
            )
            ax.add_patch(rect)

            # Add the label
            ax.text(
                x + w / 2,
                y + h / 2,
                label,
                ha="center",
                va="center",
                fontsize=16,
                color=TEXT_COLOR,
                fontweight="bold",
            )

            # Add dimension text to verify squareness
            ax.text(
                x + w / 2,
                y + h / 4,
                f"{int(w)}x{int(h)} px",
                ha="center",
                va="center",
                fontsize=11,
                color="#718096",
            )

            # Add decorative corners
            corner_len = 25
            # Top-left
            ax.plot([x, x + corner_len], [y + h, y + h], color=HIGHLIGHT_COLOR, lw=3)
            ax.plot([x, x], [y + h, y + h - corner_len], color=HIGHLIGHT_COLOR, lw=3)
            # Bottom-right
            ax.plot([x + w, x + w - corner_len], [y, y], color=HIGHLIGHT_COLOR, lw=3)
            ax.plot([x + w, x + w], [y, y + corner_len], color=HIGHLIGHT_COLOR, lw=3)

    output_path = OUTPUT_DIR / f"{name}_skeleton.png"
    plt.savefig(
        output_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor(), pad_inches=0.1
    )
    plt.close(fig)
    print(f"Generated prototype: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, boxes in LAYOUTS.items():
        draw_prototype(name, boxes)


if __name__ == "__main__":
    main()
