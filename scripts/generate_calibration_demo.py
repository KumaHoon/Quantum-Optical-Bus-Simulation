"""Generate a professor-grade calibration live-demo GIF."""

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
import matplotlib.transforms as mtrans
from matplotlib import ticker
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from quantum_optical_bus.viz_style import (
    AXIS_COLOR,
    BG_COLOR,
    PANEL_COLOR,
    SERIES_BLUE,
    SERIES_ORANGE,
    apply_ieee_style,
)


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
CONTOUR_LEVELS = 24
CAL_LABEL_SIZE = 24


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stable, professor-grade calibration demo GIF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS_DIR / "calibration_demo.gif",
        help="Output GIF path (default: assets/calibration_demo.gif).",
    )
    parser.add_argument("--n-phase1", type=int, default=36, help="Frames in pump sweep phase.")
    parser.add_argument("--n-phase2", type=int, default=24, help="Frames in loss sweep phase.")
    parser.add_argument("--fps", type=float, default=8.0, help="Output frame rate.")
    parser.add_argument("--dpi", type=int, default=92, help="Figure DPI.")
    parser.add_argument(
        "--figure-width",
        type=float,
        default=10.7,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=5.55,
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

    print(f"Pre-computing {n_phase1 + n_phase2} Wigner frames ...")
    for pump_mw, loss_db in make_sweep_schedule(n_phase1, n_phase2):
        frame = simulate_frame(float(pump_mw), float(loss_db), grid, backend)
        frames.append(frame)
        global_w_max = max(global_w_max, float(np.max(np.abs(frame.wigner))))

    return DemoData(
        frames=tuple(frames),
        xvec=grid,
        calibration_powers=calibration_powers,
        calibration_sq_db=calibration_sq_db,
        global_w_max=global_w_max,
    )


def _phase_label(frame_idx: int, n_phase1: int) -> str:
    return "Phase 1: Power sweep (intrinsic r)" if frame_idx < n_phase1 else "Phase 2: Added loss"


def configure_axes() -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    fig = plt.figure()
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[4.15, 1.25],
        width_ratios=[1.0, 1.55],
        hspace=0.20,
        wspace=0.06,
    )
    ax_dashboard = fig.add_subplot(gs[0, 0])
    ax_wigner = fig.add_subplot(gs[0, 1])
    ax_calibration = fig.add_subplot(gs[1, :])
    return fig, ax_dashboard, ax_wigner, ax_calibration


def add_transition_callout(ax: plt.Axes, *, show: bool) -> None:
    if not show:
        return
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.4, 2.4),
            9.1,
            0.68,
            boxstyle="round,pad=0.14",
            facecolor=BG_COLOR,
            edgecolor=INTRINSIC_COLOR,
            linewidth=1.1,
            alpha=0.95,
        )
    )
    ax.text(
        5.0,
        2.73,
        "Intrinsic (pre-loss) stays; observed (post-loss) degrades with loss.",
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_COLOR,
    )


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
) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor="#21262d",
            edgecolor=GRID_COLOR,
            linewidth=0.8,
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
        x - 0.15,
        y + h / 2,
        label_left,
        ha="right",
        va="center",
        fontsize=9,
        color=AXIS_COLOR,
        fontweight="bold",
    )
    ax.text(
        x + w + 0.15,
        y + h / 2,
        label_right,
        ha="left",
        va="center",
        fontsize=10,
        color=TEXT_COLOR,
        fontweight="bold",
    )


def draw_calibration_panel(
    ax: plt.Axes,
    frame: DemoFrame,
    powers: np.ndarray,
    sq_db: np.ndarray,
) -> None:
    ax.set_title("Calibration curve (intrinsic)", fontsize=10, color=TEXT_COLOR, pad=7)
    ax.set_facecolor(PANEL_COLOR)
    ax.set_xlim(0.0, 200.0)
    ax.set_xticks([0, 50, 100, 150, 200])
    y_max = float(sq_db.max())
    if y_max <= 0.0:
        y_max = 1.0
    ax.set_ylim(0.0, y_max * 1.08)
    ax.set_yticks(np.linspace(0.0, y_max, num=5))

    ax.plot(powers, sq_db, color=INTRINSIC_COLOR, lw=1.35, alpha=0.95)
    ax.scatter(
        [frame.pump_mw],
        [frame.intrinsic_sq_db],
        color=OBSERVED_COLOR,
        s=36,
        zorder=5,
    )
    ax.axvline(frame.pump_mw, color=OBSERVED_COLOR, ls="--", lw=1.0, alpha=0.9)
    ax.set_xlabel("Pump power P (mW)", fontsize=8.5, color=AXIS_COLOR, labelpad=6)
    ax.set_ylabel("Squeezing (dB)", fontsize=8.5, color=AXIS_COLOR, labelpad=6)
    ax.tick_params(axis="both", pad=3, colors=AXIS_COLOR, labelsize=8)
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
        fontsize=8,
        color=AXIS_COLOR,
    )

    ax.text(
        0.97,
        0.08,
        "Observed = loss-attenuated variance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=GRID_COLOR,
    )


def draw_dashboard(
    ax: plt.Axes,
    frame: DemoFrame,
    frame_idx: int,
    n_phase1: int,
    fig: plt.Figure,
) -> None:
    phase = _phase_label(frame_idx, n_phase1)
    phase_color = INTRINSIC_COLOR if frame_idx < n_phase1 else OBSERVED_COLOR
    show_callout = frame_idx in {n_phase1, n_phase1 + 1, n_phase1 + 2}

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.3, 8.6),
            9.3,
            1.0,
            boxstyle="round,pad=0.18",
            facecolor=phase_color,
            edgecolor="none",
            alpha=0.14,
        )
    )
    ax.text(
        4.95,
        9.1,
        phase,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=phase_color,
    )

    draw_bar(
        ax,
        3.0,
        7.4,
        5.4,
        0.58,
        frame.pump_mw / 200.0,
        INTRINSIC_COLOR,
        "Pump power",
        f"{frame.pump_mw:.0f} mW",
    )
    draw_bar(
        ax,
        3.0,
        6.45,
        5.4,
        0.58,
        frame.loss_db / 2.0,
        OBSERVED_COLOR,
        "Loss",
        f"{frame.loss_db:.2f} dB",
    )

    ax.text(
        4.9,
        5.95,
        "CALIBRATION METRICS",
        ha="center",
        fontsize=10,
        color=AXIS_COLOR,
        fontweight="bold",
    )

    ax.text(0.9, 5.15, "Squeezing parameter r", fontsize=9, color=AXIS_COLOR)
    ax.text(
        9.0,
        5.15,
        f"{frame.r:.4f}",
        fontsize=12,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="right",
    )

    label_gap = mtrans.ScaledTranslation(0, -6 / 72, fig.dpi_scale_trans)
    value_gap = mtrans.ScaledTranslation(0, -11 / 72, fig.dpi_scale_trans)

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.45, 2.5),
            9.1,
            2.0,
            boxstyle="round,pad=0.14",
            facecolor="#251f23",
            edgecolor=INTRINSIC_COLOR,
            linewidth=1.2,
        )
    )
    ax.text(
        5.0,
        4.15,
        "INTRINSIC SQUEEZING (pre-loss)",
        ha="center",
        fontsize=9,
        color=AXIS_COLOR,
        fontweight="bold",
    )
    ax.text(
        5.0,
        3.78,
        f"{frame.intrinsic_sq_db:.2f} dB",
        ha="center",
        transform=ax.transData + label_gap,
        fontsize=CAL_LABEL_SIZE,
        fontweight="bold",
        color=INTRINSIC_COLOR,
    )
    ax.text(
        5.0,
        3.25,
        "OBSERVED SQUEEZING (post-loss)",
        ha="center",
        transform=ax.transData + value_gap,
        fontsize=9,
        color=AXIS_COLOR,
        fontweight="bold",
    )
    ax.text(
        5.0,
        2.88,
        f"{frame.observed_sq_db:.2f} dB",
        transform=ax.transData + label_gap,
        ha="center",
        fontsize=CAL_LABEL_SIZE,
        fontweight="bold",
        color=OBSERVED_COLOR,
    )
    ax.text(
        0.9,
        0.8,
        "Channel transmissivity",
        fontsize=9,
        color=AXIS_COLOR,
    )

    eta = frame.transmittance
    eta_color = INTRINSIC_COLOR if eta > 0.9 else OBSERVED_COLOR if eta > 0.5 else "#8b4b53"
    ax.text(
        5.0,
        0.8,
        f"{eta:.4f}",
        fontsize=12,
        fontweight="bold",
        color=eta_color,
        ha="right",
    )

    add_transition_callout(ax, show=show_callout)


def draw_wigner_panel(
    ax: plt.Axes,
    frame: DemoFrame,
    xvec: np.ndarray,
    levels: np.ndarray,
) -> None:
    ax.contourf(
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
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        linestyle="--",
        color=AXIS_COLOR,
        linewidth=1.0,
        alpha=0.8,
    )
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (position quadrature)", fontsize=9, color=TEXT_COLOR, labelpad=9)
    ax.set_ylabel("p (momentum quadrature)", fontsize=9, color=TEXT_COLOR, labelpad=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.set_title(
        f"Wigner function: r = {frame.r:.3f}",
        fontsize=10,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=8,
    )
    ax.tick_params(labelsize=8, pad=3, colors=AXIS_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_alpha(0.9)
    ax.grid(alpha=0.12, color=GRID_COLOR)


def optimize_gif(tmp_path: Path, output: Path, *, colors: int, fps: float) -> None:
    with Image.open(tmp_path) as im:
        frames = []
        for frame in range(im.n_frames):
            im.seek(frame)
            rgb = im.convert("RGB")
            frames.append(rgb.quantize(colors=colors, dither=Image.Dither.NONE))

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
    tmp_path.unlink(missing_ok=True)
    print(f"[OK] Saved optimized GIF to {output} ({output.stat().st_size} bytes)")


def save_mp4_if_available(gif_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[INFO] ffmpeg not found; skipping MP4 generation.")
        return

    try:
        import imageio.v2 as imageio

        with imageio.get_reader(gif_path) as reader:
            frames = [f for f in reader]
        if not frames:
            print("[WARN] GIF has no frames; skipped MP4.")
            return
        imageio.mimsave(gif_path.with_suffix(".mp4"), frames, fps=8)
        print(f"[OK] Also saved {gif_path.with_suffix('.mp4')}")
    except Exception as exc:
        print(f"[WARN] MP4 generation failed: {exc}")


def run_animation(data: DemoData, config: RenderConfig) -> None:
    apply_ieee_style(base_font_size=10, tick_font_size=9, dpi=config.dpi)
    levels = np.linspace(
        -max(data.global_w_max, 1e-6),
        max(data.global_w_max, 1e-6),
        CONTOUR_LEVELS,
    )
    fig, ax_dash, ax_wig, ax_cal = configure_axes()
    fig.set_size_inches(config.figure_width, config.figure_height)
    fig.patch.set_facecolor(BG_COLOR)

    # Center the bottom calibration axis under both top panels.
    cal_pos = ax_cal.get_position()
    cal_width = cal_pos.width * 0.72
    cal_x0 = 0.5 - cal_width / 2.0
    ax_cal.set_position((cal_x0, cal_pos.y0, cal_width, cal_pos.height))

    fig.subplots_adjust(top=0.95, bottom=0.07, left=0.05, right=0.985, hspace=0.20, wspace=0.06)

    fig.suptitle(
        "Real-time Calibration Simulation",
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.985,
    )

    def draw_frame(frame_idx: int) -> None:
        frame = data.frames[frame_idx]
        ax_dash.cla()
        ax_wig.cla()
        ax_cal.cla()
        draw_dashboard(ax_dash, frame, frame_idx, config.n_phase1, fig)
        draw_wigner_panel(ax_wig, frame, data.xvec, levels)
        draw_calibration_panel(ax_cal, frame, data.calibration_powers, data.calibration_sq_db)

    print(f"Rendering {len(data.frames)} frames at {config.fps:.1f} fps ...")
    anim = FuncAnimation(fig, draw_frame, frames=len(data.frames), blit=False)
    tmp_path = config.output.with_suffix(".tmp.gif")
    anim.save(tmp_path, writer=PillowWriter(fps=config.fps), dpi=config.dpi)
    plt.close(fig)
    optimize_gif(tmp_path, config.output, colors=config.gif_colors, fps=config.fps)

    if config.save_mp4:
        save_mp4_if_available(config.output)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = build_demo_data(args.n_phase1, args.n_phase2)
    render_cfg = RenderConfig(
        output=args.output,
        n_phase1=args.n_phase1,
        fps=args.fps,
        dpi=args.dpi,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        gif_colors=args.gif_colors,
        save_mp4=args.save_mp4,
    )
    run_animation(data, render_cfg)


if __name__ == "__main__":
    main()
