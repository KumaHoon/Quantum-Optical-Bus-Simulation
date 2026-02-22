"""Generate a review-grade calibration live-demo GIF."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

try:
    from figstyle import (
        apply_style,
        canonical_canvas_inches,
        canonical_dpi,
        write_figure_meta,
    )
except ModuleNotFoundError:
    from scripts.figstyle import (
        apply_style,
        canonical_canvas_inches,
        canonical_dpi,
        write_figure_meta,
    )
from quantum_optical_bus.viz_style_ieee import (
    AXIS_COLOR,
    AXIS_LABEL_FONT_SIZE,
    AXIS_TITLE_FONT_SIZE,
    SMALL_FONT_SIZE,
    BG_COLOR,
    PANEL_COLOR,
    SERIES_BLUE,
    SERIES_ORANGE,
    compact_axis_formatter,
    set_tab_title,
)
try:
    from generate_prototypes import apply_prototype
except ModuleNotFoundError:
    from scripts.generate_prototypes import apply_prototype
try:
    from asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs
except ModuleNotFoundError:
    from scripts.asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SRC_ROOT / "src"))


@dataclass(frozen=True)
class QuantumBackend:
    calculate_squeezing: Callable[[float], float]
    db_to_eta: Callable[[float], float]
    observed_squeezing_from_cov: Callable[[Any], tuple[float, float]]
    sf_cov_to_vacuum05: Callable[[Any], Any]
    program_cls: Callable[[int], Any]
    engine_cls: Callable[[str], Any]
    sgate_cls: Callable[[float], Any]
    loss_cls: Callable[[float], Any]


def load_quantum_backend() -> QuantumBackend:
    """Load simulation dependencies after path setup."""
    import quantum_optical_bus.compat  # noqa: F401
    import strawberryfields as sf
    from strawberryfields.ops import LossChannel, Sgate
    from quantum_optical_bus.interface import calculate_squeezing
    from quantum_optical_bus.units import (
        db_to_eta,
        observed_squeezing_from_cov,
        sf_cov_to_vacuum05,
    )

    return QuantumBackend(
        calculate_squeezing=calculate_squeezing,
        db_to_eta=db_to_eta,
        observed_squeezing_from_cov=observed_squeezing_from_cov,
        sf_cov_to_vacuum05=sf_cov_to_vacuum05,
        program_cls=sf.Program,
        engine_cls=sf.Engine,
        sgate_cls=Sgate,
        loss_cls=LossChannel,
    )


ASSETS_DIR = SRC_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

INTRINSIC_COLOR = SERIES_BLUE
OBSERVED_COLOR = SERIES_ORANGE
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#30363d"
CONTOUR_LEVELS = 18
CAL_LABEL_SIZE = 20
WIGNER_LABELPAD = 11
WIGNER_CIRCLE_LINEWIDTH = 0.18
WIGNER_CIRCLE_ALPHA = 0.95
CALIBRATION_DEMO_TITLE = "Real-Time Calibration Simulation"
CALIBRATION_DEMO_FOOTNOTE = (
    "Calibration Demo Scenario | Phase sweeps over operating points\n"
    "Demonstrates live feedback loops tracking maximum quadrature squeezing margins.\n"
    "Observed: loss-attenuated variance."
)


def _format_db(value: float) -> str:
    if abs(value) < 5e-4:
        value = 0.0
    else:
        value = float(value)
    return f"{value:.2f}"


def _refresh_theme_colors() -> None:
    """Refresh module color constants after applying an output style."""

    global BG_COLOR, PANEL_COLOR, GRID_COLOR, TEXT_COLOR
    BG_COLOR = str(plt.rcParams.get("figure.facecolor", BG_COLOR))
    PANEL_COLOR = str(plt.rcParams.get("axes.facecolor", PANEL_COLOR))
    GRID_COLOR = str(plt.rcParams.get("grid.color", GRID_COLOR))
    TEXT_COLOR = str(plt.rcParams.get("text.color", TEXT_COLOR))


@dataclass(frozen=True)
class DemoFrame:
    wigner: np.ndarray
    r: float
    intrinsic_sq_db: float
    observed_sq_db: float
    pump_mw: float
    loss_db: float
    transmittance: float


@dataclass(frozen=True)
class DemoData:
    frames: tuple[DemoFrame, ...]
    xvec: np.ndarray
    calibration_powers: np.ndarray
    calibration_sq_db: np.ndarray
    global_w_max: float


@dataclass(frozen=True)
class RenderConfig:
    output: Path
    n_phase1: int
    fps: float
    dpi: int
    figure_width: float
    figure_height: float
    gif_colors: int
    save_mp4: bool


def _contract_canvas_for(profile: str) -> tuple[float, float]:
    return canonical_canvas_inches(profile)


def _contract_dpi(profile: str) -> int:
    return canonical_dpi(profile)


def _write_demo_metadata(path: Path, profile: str, notes: str) -> None:
    dpi = _contract_dpi(profile)
    width_in, height_in = _contract_canvas_for(profile)
    write_figure_meta(
        path,
        figure_id=path.stem,
        profile=profile,
        generator_script="scripts/generate_calibration_demo.py",
        generator_args=(f"--output-dir={path.parent}", f"--profile={profile}"),
        labels={
            "title": "Calibration demo (intrinsic vs observed squeezing)",
            "xlabel": "Pump power P [mW] / loss [dB]",
            "ylabel": "Squeezing [dB] / variance [SNU]",
        },
        units={"x": "mW/dB", "y": "dB/SNU", "time": "frame"},
        notes=notes,
        seed=11,
        dpi=dpi,
        canvas_px=(int(round(width_in * dpi)), int(round(height_in * dpi))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a stable calibration demo GIF.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS_DIR / "calibration_demo.gif",
        help="Output GIF path (default: assets/calibration_demo.gif).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory where output GIF is written.",
    )
    parser.add_argument("--n-phase1", type=int, default=36, help="Frames in pump sweep phase.")
    parser.add_argument("--n-phase2", type=int, default=24, help="Frames in loss sweep phase.")
    parser.add_argument("--fps", type=float, default=8.0, help="Output frame rate.")
    parser.add_argument("--dpi", type=int, default=canonical_dpi("web"), help="Figure DPI.")
    default_web_width, default_web_height = _contract_canvas_for("web")
    parser.add_argument(
        "--figure-width",
        type=float,
        default=default_web_width,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=default_web_height,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--gif-colors",
        type=int,
        default=144,
        help="GIF palette colors (128-160 is usually sufficient).",
    )
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Also write an assets/calibration_demo.mp4 when ffmpeg is available.",
    )
    parser.add_argument(
        "--profile",
        default="web",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    return parser.parse_args()


def make_sweep_schedule(n_phase1: int, n_phase2: int) -> Iterable[tuple[float, float]]:
    powers = np.concatenate([np.linspace(0, 200, n_phase1), np.full(n_phase2, 200.0)])
    losses_db = np.concatenate([np.zeros(n_phase1), np.linspace(0, 2.0, n_phase2)])
    return zip(powers, losses_db)


def simulate_frame(
    pump_mw: float,
    loss_db: float,
    xvec: np.ndarray,
    backend: QuantumBackend,
) -> DemoFrame:
    r = backend.calculate_squeezing(float(pump_mw))
    intrinsic_sq_db = -10 * np.log10(np.exp(-2 * r)) if r > 0 else 0.0
    transmittance = float(backend.db_to_eta(float(loss_db)))

    program = backend.program_cls(1)
    with program.context as q:
        if r > 0:
            backend.sgate_cls(r) | q[0]
        if transmittance < 1.0:
            backend.loss_cls(transmittance) | q[0]

    result = backend.engine_cls("gaussian").run(program)
    wigner = result.state.wigner(0, xvec, xvec)
    cov = backend.sf_cov_to_vacuum05(result.state.cov())
    observed_sq_db, _ = backend.observed_squeezing_from_cov(cov)

    return DemoFrame(
        wigner=wigner,
        r=r,
        intrinsic_sq_db=intrinsic_sq_db,
        observed_sq_db=observed_sq_db,
        pump_mw=pump_mw,
        loss_db=loss_db,
        transmittance=transmittance,
    )


def build_demo_data(n_phase1: int, n_phase2: int) -> DemoData:
    grid = np.linspace(-5.0, 5.0, 100)
    assert n_phase1 + n_phase2 >= 3, "Need at least 3 frames for animation"
    backend = load_quantum_backend()
    frames: list[DemoFrame] = []
    calibration_powers = np.linspace(0.0, 200.0, 160)
    calibration_sq_db_list: list[float] = []
    for pump in calibration_powers:
        if pump > 0:
            intrinsic = backend.calculate_squeezing(float(pump))
            calibration_sq_db_list.append(-10 * np.log10(np.exp(-2.0 * intrinsic)))
        else:
            calibration_sq_db_list.append(0.0)
    calibration_sq_db = np.array(calibration_sq_db_list, dtype=float)
    global_w_max = 0.0

    for pump_mw, loss_db in make_sweep_schedule(n_phase1, n_phase2):
        frame = simulate_frame(float(pump_mw), float(loss_db), grid, backend)
        frames.append(frame)
        global_w_max = max(global_w_max, float(np.max(np.abs(frame.wigner))))

    assert len(frames) >= 3, "Demo must contain at least 3 frames."
    return DemoData(
        frames=tuple(frames),
        xvec=grid,
        calibration_powers=calibration_powers,
        calibration_sq_db=calibration_sq_db,
        global_w_max=global_w_max,
    )


def _phase_label(frame_idx: int, n_phase1: int) -> str:
    return "Phase 1: Power sweep (intrinsic r)" if frame_idx < n_phase1 else "Phase 2: Added loss"


def configure_axes(profile: str) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig = plt.figure()
    if profile in ("web", "paper"):
        box_axes = apply_prototype(
            fig,
            "calibration_demo",
            profile,
            custom_title=CALIBRATION_DEMO_TITLE if profile == "web" else "",
            custom_footnote=CALIBRATION_DEMO_FOOTNOTE,
            hide_layout=False,
        )
        ax_dashboard = fig.add_axes(box_axes["Dashboard Metrics"])
        ax_wigner = fig.add_axes(box_axes["Wigner Function"])
        ax_calibration = fig.add_axes(box_axes["Calibration Sweep Phase"])
    else:
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=[4.45, 1.4],
            width_ratios=[1.0, 0.72],
            hspace=0.30,
            wspace=0.28,
        )
        ax_dashboard = fig.add_subplot(gs[0, 0])
        ax_wigner = fig.add_subplot(gs[0, 1])
        cal_center = gs[1, :].subgridspec(
            ncols=3,
            nrows=1,
            width_ratios=[0.15, 0.70, 0.15],
            wspace=0.0,
        )
        ax_calibration = fig.add_subplot(cal_center[0, 1])
    return fig, ax_dashboard, ax_wigner, ax_calibration


def draw_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    frac: float,
    color: str,
    label_left: str,
    label_right: str,
    *,
    is_paper: bool = False,
) -> None:
    text_size = float(plt.rcParams.get("axes.labelsize", SMALL_FONT_SIZE))
    text_size = max(6.0, text_size - (4.0 if is_paper else 0.0))
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor="#21262d",
            edgecolor=GRID_COLOR,
            linewidth=0.75,
        )
    )
    if frac > 0:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                max(w * frac, 0.08),
                h,
                boxstyle="round,pad=0.08",
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
            )
        )
    ax.text(
        x - 0.45,
        y + h / 2,
        label_left,
        ha="right",
        va="center",
        fontsize=text_size + 0.3,
        color=AXIS_COLOR,
        fontweight="bold",
    )
    ax.text(
        x + w + 0.45,
        y + h / 2,
        label_right,
        ha="left",
        va="center",
        fontsize=text_size + 0.3,
        color=TEXT_COLOR,
        fontweight="bold",
    )


def _data_x_padding_for_pixels(ax: plt.Axes, px: float) -> float:
    fig = ax.figure
    if fig is None:
        return 0.0
    axis_width = ax.get_position().width * float(fig.get_figwidth())
    axis_width_px = max(1.0, axis_width * float(fig.dpi))
    x_min, x_max = ax.get_xlim()
    return px * (float(x_max) - float(x_min)) / axis_width_px


def _points_for_pixels(fig: plt.Figure, px: float) -> float:
    dpi = float(getattr(fig, "dpi", 100.0) or 100.0)
    return 72.0 * px / dpi


def _data_y_padding_for_pixels(ax: plt.Axes, px: float) -> float:
    fig = ax.figure
    if fig is None:
        return 0.0
    axis_height = ax.get_position().height * float(fig.get_figheight())
    axis_height_px = max(1.0, axis_height * float(fig.dpi))
    y_min, y_max = ax.get_ylim()
    return px * (float(y_max) - float(y_min)) / axis_height_px


def draw_calibration_panel(
    ax: plt.Axes,
    frame: DemoFrame,
    powers: np.ndarray,
    sq_db: np.ndarray,
    *,
    is_paper: bool = False,
) -> None:
    size_delta = 4.0 if is_paper else 0.0
    title_size = max(7.0, float(plt.rcParams.get("axes.titlesize", AXIS_TITLE_FONT_SIZE)) - size_delta)
    label_size = max(6.0, float(plt.rcParams.get("axes.labelsize", AXIS_LABEL_FONT_SIZE)) - size_delta)
    tick_size = max(5.0, float(plt.rcParams.get("xtick.labelsize", AXIS_LABEL_FONT_SIZE)) - size_delta)
    ax.set_title("Calibration curve", fontsize=title_size, color=TEXT_COLOR, pad=7)
    ax.set_facecolor(PANEL_COLOR)
    ax.set_xlim(0.0, 200.0)
    ax.set_xticks([0, 50, 100, 150, 200])
    y_max = float(max(1.0, sq_db.max()))
    ax.set_ylim(0.0, y_max * 1.05)
    ax.set_yticks(np.linspace(0.0, y_max, num=5))

    ax.plot(powers, sq_db, color=INTRINSIC_COLOR, lw=0.95, alpha=0.95)
    ax.scatter(
        [frame.pump_mw],
        [frame.intrinsic_sq_db],
        color=OBSERVED_COLOR,
        s=36,
        zorder=5,
    )
    ax.axvline(frame.pump_mw, color=OBSERVED_COLOR, ls="--", lw=0.85, alpha=0.9)
    ax.set_xlabel("Pump power P (mW)", fontsize=label_size, color=AXIS_COLOR, labelpad=6)
    ax.set_ylabel("Squeezing (dB)", fontsize=label_size, color=AXIS_COLOR, labelpad=6)
    ax.tick_params(axis="both", pad=3, colors=AXIS_COLOR, labelsize=tick_size)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.grid(alpha=0.23, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.text(
        0.03,
        0.95,
        f"Current operating point: P={frame.pump_mw:.1f} mW",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=label_size,
        color=AXIS_COLOR,
    )

def draw_dashboard(
    ax: plt.Axes,
    frame: DemoFrame,
    frame_idx: int,
    n_phase1: int,
    *,
    is_paper: bool = False,
) -> None:
    label_size = float(plt.rcParams.get("axes.labelsize", AXIS_LABEL_FONT_SIZE))
    title_size = float(plt.rcParams.get("axes.titlesize", AXIS_TITLE_FONT_SIZE))
    size_delta = 4.0 if is_paper else 0.0
    phase_font = max(9.0, title_size - size_delta)
    metrics_label_font = max(6.0, label_size - 1.0 - size_delta)
    metrics_value_font = max(6.0, label_size + 0.8 - size_delta)
    phase = _phase_label(frame_idx, n_phase1)
    phase_color = INTRINSIC_COLOR if frame_idx < n_phase1 else OBSERVED_COLOR

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    phase_pad_px = 10.0 if is_paper else 0.0
    label_shift_px = 15.0 if is_paper else 0.0
    value_shift_px = 10.0 if is_paper else 0.0
    bar_shift_y_px = 40.0 if is_paper else 0.0

    pad_data = _data_x_padding_for_pixels(ax, phase_pad_px)
    phase_x = max(0.0, 0.3 - pad_data)
    phase_w = 9.3 + (2.0 * pad_data)
    bar_shift = _data_y_padding_for_pixels(ax, bar_shift_y_px)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (phase_x, 9.05),
            phase_w,
            0.63,
            boxstyle="round,pad=0.18",
            facecolor=phase_color,
            edgecolor="none",
            alpha=0.14,
        )
    )
    label_offset_pts = _points_for_pixels(ax.figure, label_shift_px) if ax.figure is not None else 0.0
    value_offset_pts = _points_for_pixels(ax.figure, value_shift_px) if ax.figure is not None else 0.0
    ax.text(
        4.95,
        9.36,
        phase,
        ha="center",
        va="center",
        fontsize=phase_font,
        fontweight="bold",
        color=phase_color,
    )
    draw_bar(
        ax,
        3.0,
        7.4 - bar_shift,
        5.4,
        0.58,
        min(frame.pump_mw / 200.0, 1.0),
        INTRINSIC_COLOR,
        "Pump power",
        f"{frame.pump_mw:.0f} mW",
        is_paper=is_paper,
    )
    draw_bar(
        ax,
        3.0,
        6.45 - bar_shift,
        5.4,
        0.58,
        frame.loss_db / 2.0,
        OBSERVED_COLOR,
        "Loss",
        f"{frame.loss_db:.2f} dB",
        is_paper=is_paper,
    )

    metric_label_x = -0.01
    metric_value_x = 1.01
    eta = frame.transmittance
    eta_color = INTRINSIC_COLOR if eta > 0.9 else OBSERVED_COLOR if eta > 0.5 else "#8b4b53"
    metrics_anchor_y = 0.30
    metrics_anchor_shift_px = -105.0
    metrics_anchor_shift_pts = metrics_anchor_shift_px * 72.0 / float(
        ax.figure.dpi if ax.figure and ax.figure.dpi else 72.0
    )
    label_font = metrics_label_font
    value_font = metrics_value_font
    row_gap_pts = 46.0 * 72.0 / float(ax.figure.dpi if ax.figure and ax.figure.dpi else 72.0)
    value_x_shift_px = 0.0

    metrics_rows = [
        ("Squeezing parameter r (dimensionless)", f"{frame.r:.4f}", TEXT_COLOR),
        ("INTRINSIC squeezing (pre-loss, dB)", f"{_format_db(frame.intrinsic_sq_db)} dB", INTRINSIC_COLOR),
        ("OBSERVED squeezing (post-loss, dB)", f"{_format_db(max(frame.observed_sq_db, 0.0))} dB", OBSERVED_COLOR),
        ("Channel transmissivity (η)", f"{eta:.4f}", eta_color),
    ]

    for row_idx, (label, value, value_color) in enumerate(metrics_rows):
        row_offset = (len(metrics_rows) - 1 - row_idx) * row_gap_pts
        ax.annotate(
            label,
            xy=(metric_label_x, metrics_anchor_y),
            xycoords="axes fraction",
            xytext=(-label_offset_pts, row_offset + metrics_anchor_shift_pts),
            textcoords="offset points",
            transform=ax.transAxes,
            fontsize=label_font,
            color=AXIS_COLOR,
            ha="left",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )
        ax.annotate(
            value,
            xy=(metric_value_x, metrics_anchor_y),
            xycoords="axes fraction",
            xytext=(value_offset_pts, row_offset + metrics_anchor_shift_pts),
            textcoords="offset points",
            transform=ax.transAxes,
            fontsize=value_font,
            fontweight="bold",
            color=value_color,
            ha="right",
            va="bottom",
            clip_on=False,
        )


def draw_wigner_panel(
    ax: plt.Axes,
    frame: DemoFrame,
    xvec: np.ndarray,
    levels: np.ndarray,
    *,
    is_paper: bool = False,
) -> None:
    size_delta = 4.0 if is_paper else 0.0
    label_size = max(6.0, float(plt.rcParams.get("axes.labelsize", AXIS_LABEL_FONT_SIZE)) - size_delta)
    title_size = max(7.0, float(plt.rcParams.get("axes.titlesize", AXIS_TITLE_FONT_SIZE)) - size_delta)
    tick_size = max(5.0, float(plt.rcParams.get("xtick.labelsize", AXIS_LABEL_FONT_SIZE)) - size_delta)
    prev_cbar = getattr(ax, "_qob_wigner_colorbar", None)
    if prev_cbar is not None:
        try:
            prev_cbar.remove()
        except Exception:
            pass
        setattr(ax, "_qob_wigner_colorbar", None)
    prev_cax = getattr(ax, "_qob_wigner_cax", None)
    if prev_cax is not None:
        try:
            prev_cax.remove()
        except Exception:
            pass
        setattr(ax, "_qob_wigner_cax", None)
    cset = ax.contourf(
        xvec,
        xvec,
        frame.wigner,
        levels=levels,
        cmap="RdBu_r",
        vmin=levels.min(),
        vmax=levels.max(),
        antialiased=True,
        extend="both",
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    # Circle removed per review request
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal", adjustable="box")
    try:
        ax.set_box_aspect(1.0)
    except Exception:
        pass
    ax.set_xlabel(
        "x quadrature",
        fontsize=label_size,
        color=TEXT_COLOR,
        labelpad=max(6, WIGNER_LABELPAD - 3),
    )
    ax.set_ylabel(
        "p quadrature",
        fontsize=label_size,
        color=TEXT_COLOR,
        labelpad=7,
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.set_title(
        f"Wigner: r = {frame.r:.3f}",
        fontsize=title_size,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=8,
    )
    ax.tick_params(labelsize=tick_size, pad=2, colors=AXIS_COLOR)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.2%", pad=0.08)
    cbar = ax.figure.colorbar(cset, cax=cax)
    cbar.set_label("Wigner amplitude", fontsize=label_size)
    cbar.ax.tick_params(labelsize=max(7.0, tick_size - 1.3))
    cbar.ax.yaxis.set_major_formatter(compact_axis_formatter())
    setattr(ax, "_qob_wigner_colorbar", cbar)
    setattr(ax, "_qob_wigner_cax", cax)
    ax.xaxis.set_major_formatter(compact_axis_formatter())
    ax.yaxis.set_major_formatter(compact_axis_formatter())
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_alpha(0.9)
    ax.grid(alpha=0.12, color=GRID_COLOR)


def optimize_gif(tmp_path: Path, output: Path, *, colors: int, fps: float) -> None:
    gif_colors = max(128, min(160, colors))
    with Image.open(tmp_path) as im:
        frames = []
        for frame in range(im.n_frames):
            im.seek(frame)
            rgb = im.convert("RGB")
            frames.append(rgb.quantize(colors=gif_colors, dither=Image.Dither.NONE))

        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            loop=im.info.get("loop", 0),
            duration=int(1000 / fps),
            optimize=True,
            disposal=2,
            include_color_table=True,
        )
    try:
        tmp_path.unlink(missing_ok=True)
    except PermissionError:
        # Pillow on Windows can keep the temporary file handle briefly after writer close.
        # Keep cleanup best-effort to avoid hard failure in CI/Windows environments.
        print(f"[WARN] Temporary GIF {tmp_path} is still locked; leaving it for manual cleanup.")
    print(f"[OK] Saved optimized GIF to {output} ({output.stat().st_size} bytes)")


def save_mp4_if_available(gif_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[INFO] ffmpeg not found; skipping MP4 generation.")
        return

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[WARN] imageio unavailable: {exc}")
        return

    with imageio.get_reader(gif_path) as reader:
        frames = [f for f in reader]
    if not frames:
        print("[WARN] GIF has no frames; skipped MP4.")
        return
    imageio.mimsave(gif_path.with_suffix(".mp4"), frames, fps=8)
    print(f"[OK] Also saved {gif_path.with_suffix('.mp4')}")


def run_animation(data: DemoData, config: RenderConfig, profile: str) -> None:
    contract_dpi = _contract_dpi(profile)
    figure_width, figure_height = _contract_canvas_for(profile)
    apply_style(
        profile,
        base_font_size=11 if profile == "paper" else 10,
        tick_font_size=10 if profile == "paper" else 9,
        dpi=contract_dpi,
    )
    _refresh_theme_colors()

    levels = np.linspace(
        -max(data.global_w_max, 1e-6),
        max(data.global_w_max, 1e-6),
        CONTOUR_LEVELS,
    )
    fig, ax_dash, ax_wig, ax_cal = configure_axes(profile)
    fig.set_size_inches(figure_width, figure_height)
    fig.patch.set_facecolor(plt.rcParams.get("figure.facecolor", "#0d1117"))
    if profile != "web":
        set_tab_title(fig, CALIBRATION_DEMO_TITLE, mode=profile)

    def draw_frame(frame_idx: int) -> None:
        frame = data.frames[frame_idx]
        ax_dash.cla()
        ax_wig.cla()
        ax_cal.cla()
        draw_dashboard(
            ax_dash,
            frame,
            frame_idx,
            config.n_phase1,
            is_paper=(profile == "paper"),
        )
        draw_wigner_panel(
            ax_wig,
            frame,
            data.xvec,
            levels,
            is_paper=(profile == "paper"),
        )
        draw_calibration_panel(
            ax_cal,
            frame,
            data.calibration_powers,
            data.calibration_sq_db,
            is_paper=(profile == "paper"),
        )

    print(f"Rendering {len(data.frames)} frames at {config.fps:.1f} fps ...")
    anim = FuncAnimation(fig, draw_frame, frames=len(data.frames), blit=False)
    tmp_path = config.output.with_suffix(".tmp.gif")
    anim.save(tmp_path, writer=PillowWriter(fps=config.fps), dpi=contract_dpi)
    plt.close(fig)

    outputs = resolve_outputs(config.output, profile)
    primary = outputs[0]
    print(f"[INFO] Output targets: {[str(path) for path in outputs]}")
    optimize_gif(tmp_path, primary, colors=config.gif_colors, fps=config.fps)
    if len(outputs) > 1:
        for extra in outputs[1:]:
            extra.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(primary, extra)
    metadata_notes = (
        "Calibration demo validates intrinsic vs observed squeezing and monotonic loss behavior "
        "across pump and loss sweeps."
    )
    for output_path in outputs:
        _write_demo_metadata(path=output_path, profile=profile, notes=metadata_notes)
    if config.save_mp4:
        save_mp4_if_available(primary)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert args.n_phase1 + args.n_phase2 >= 3, "Need at least 3 frames to create animation"

    if args.output.parent == Path("."):
        args.output = args.output_dir / args.output.name

    data = build_demo_data(args.n_phase1, args.n_phase2)
    profiles = normalize_profiles(args.profile)

    for profile in profiles:
        cfg_output = args.output
        cfg = RenderConfig(
            output=cfg_output,
            n_phase1=args.n_phase1,
            fps=args.fps,
            dpi=args.dpi,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
            gif_colors=args.gif_colors,
            save_mp4=args.save_mp4,
        )
        run_animation(data, cfg, profile)


if __name__ == "__main__":
    main()
