"""Generate a stable, readable calibration demo GIF for README presentation."""

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
from PIL import Image


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SRC_ROOT / "src"))


@dataclass(frozen=True)
class QuantumBackend:
    calculate_squeezing: Callable[[float], float]
    db_to_eta: Callable[[float], float]
    observed_squeezing_from_cov: Callable[[Any], tuple[float, Any]]
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


BG = "#f5f5f5"
PANEL = "#ffffff"
ACCENT = "#1a73e8"
RED = "#c62828"
GREEN = "#188038"
ORANGE = "#e37400"
GRAY = "#5f6368"
DARK = "#202124"
LIGHT_BRD = "#dadce0"


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
    global_w_max: float


@dataclass(frozen=True)
class RenderConfig:
    output: Path
    n_phase1: int
    fps: float
    dpi: int
    figure_width: float
    figure_height: float
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
    parser.add_argument("--n-phase1", type=int, default=40, help="Frames in pump sweep phase.")
    parser.add_argument("--n-phase2", type=int, default=30, help="Frames in loss sweep phase.")
    parser.add_argument("--fps", type=float, default=8.0, help="Output frame rate.")
    parser.add_argument("--dpi", type=int, default=100, help="Figure DPI.")
    parser.add_argument("--figure-width", type=float, default=12.0, help="Figure width in inches.")
    parser.add_argument("--figure-height", type=float, default=5.0, help="Figure height in inches.")
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
    grid = np.linspace(-5.0, 5.0, 120)
    backend = load_quantum_backend()
    frames: list[DemoFrame] = []
    global_w_max = 0.0

    print(f"Pre-computing {n_phase1 + n_phase2} Wigner frames ...")
    for pump_mw, loss_db in make_sweep_schedule(n_phase1, n_phase2):
        frame = simulate_frame(float(pump_mw), float(loss_db), grid, backend)
        frames.append(frame)
        global_w_max = max(global_w_max, float(np.max(np.abs(frame.wigner))))

    print("Done.")
    return DemoData(frames=tuple(frames), xvec=grid, global_w_max=global_w_max)


def _phase_label(frame_idx: int, n_phase1: int) -> str:
    return "Phase 1: Power Sweep" if frame_idx < n_phase1 else "Phase 2: Propagation Loss"


def configure_axes() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": LIGHT_BRD,
            "axes.labelcolor": DARK,
            "text.color": DARK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "font.family": "sans-serif",
            "font.size": 12,
        }
    )
    fig = plt.figure()
    ax_dashboard = fig.add_axes([0.025, 0.08, 0.36, 0.82])
    ax_wigner = fig.add_axes([0.42, 0.08, 0.56, 0.82])
    return fig, ax_dashboard, ax_wigner


def add_transition_callout(ax: plt.Axes, *, show: bool) -> None:
    if not show:
        return
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.4, 2.4),
            9.2,
            0.7,
            boxstyle="round,pad=0.14",
            facecolor="#e8f0fe",
            edgecolor=ACCENT,
            linewidth=1.2,
            alpha=0.95,
        )
    )
    ax.text(
        5.0,
        2.75,
        "Intrinsic (pre-loss) stays; Observed (post-loss) decreases with loss.",
        ha="center",
        va="center",
        fontsize=10,
        color=DARK,
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
            facecolor="#f1f3f4",
            edgecolor=LIGHT_BRD,
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
                alpha=0.88,
            )
        )
    ax.text(
        x - 0.15,
        y + h / 2,
        label_left,
        ha="right",
        va="center",
        fontsize=12,
        color=GRAY,
        fontweight="bold",
    )
    ax.text(
        x + w + 0.15,
        y + h / 2,
        label_right,
        ha="left",
        va="center",
        fontsize=12,
        color=DARK,
        fontweight="bold",
    )


def draw_dashboard(ax: plt.Axes, frame: DemoFrame, frame_idx: int, n_phase1: int) -> None:
    phase = _phase_label(frame_idx, n_phase1)
    phase_color = ACCENT if frame_idx < n_phase1 else ORANGE
    show_callout = frame_idx in {n_phase1, n_phase1 + 1, n_phase1 + 2}

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.3, 8.6),
            9.2,
            1.0,
            boxstyle="round,pad=0.18",
            facecolor=phase_color,
            edgecolor="none",
            alpha=0.12,
        )
    )
    ax.text(
        4.9,
        9.1,
        phase,
        ha="center",
        va="center",
        fontsize=15,
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
        ACCENT,
        "Pump Power",
        f"{frame.pump_mw:.0f} mW",
    )
    draw_bar(
        ax, 3.0, 6.5, 5.4, 0.58, frame.loss_db / 2.0, ORANGE, "Loss", f"{frame.loss_db:.2f} dB"
    )

    ax.text(
        4.9,
        5.83,
        "CALIBRATION METRICS",
        ha="center",
        fontsize=12,
        color=GRAY,
        fontweight="bold",
    )

    ax.text(0.8, 5.0, "Squeezing parameter r:", fontsize=12, color=GRAY)
    ax.text(
        9.1,
        5.0,
        f"{frame.r:.4f}",
        fontsize=15,
        fontweight="bold",
        color=GREEN,
        ha="right",
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.45, 2.8),
            9.1,
            2.0,
            boxstyle="round,pad=0.16",
            facecolor="#fce8e6",
            edgecolor=RED,
            linewidth=1.5,
        )
    )
    ax.text(
        5.0,
        4.45,
        "INTRINSIC SQUEEZING (pre-loss)",
        ha="center",
        fontsize=10,
        color=GRAY,
        fontweight="bold",
    )
    ax.text(
        5.0,
        3.95,
        f"{frame.intrinsic_sq_db:.2f} dB",
        ha="center",
        fontsize=28,
        fontweight="bold",
        color=GRAY,
    )
    ax.text(
        5.0,
        3.45,
        "OBSERVED SQUEEZING (post-loss)",
        ha="center",
        fontsize=10,
        color=RED,
        fontweight="bold",
    )
    ax.text(
        5.0,
        2.85,
        f"{frame.observed_sq_db:.2f} dB",
        ha="center",
        fontsize=28,
        fontweight="bold",
        color=RED,
    )

    add_transition_callout(ax, show=show_callout)

    ax.text(
        0.8,
        1.20,
        "Transmissivity",
        fontsize=11,
        color=GRAY,
    )
    eta = frame.transmittance
    eta_color = GREEN if eta > 0.9 else ORANGE if eta > 0.5 else RED
    ax.text(
        5.0,
        1.2,
        f"{eta:.4f}",
        fontsize=14,
        fontweight="bold",
        color=eta_color,
        ha="right",
    )


def draw_wigner_panel(ax: plt.Axes, frame: DemoFrame, xvec: np.ndarray, levels: np.ndarray) -> None:
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
    ax.contour(
        xvec,
        xvec,
        frame.wigner,
        levels=20,
        colors="k",
        linewidths=0.12,
        alpha=0.35,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        linestyle="--",
        color="#555555",
        linewidth=1.1,
        alpha=0.8,
        label="Vacuum reference",
    )
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x (position)", fontsize=12)
    ax.set_ylabel("p (momentum)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.set_title(
        f"Wigner: r = {frame.r:.3f}",
        fontsize=13,
        fontweight="bold",
        color=DARK,
        pad=8,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor=LIGHT_BRD)
    ax.grid(alpha=0.12, color=GRAY)


def optimize_gif(tmp_path: Path, output: Path) -> None:
    with Image.open(tmp_path) as im:
        frames = []
        for frame in range(im.n_frames):
            im.seek(frame)
            rgb = im.convert("RGB")
            frames.append(rgb.quantize(colors=180, dither=Image.Dither.NONE))

        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            loop=im.info.get("loop", 0),
            duration=im.info.get("duration", 125),
            optimize=True,
            disposal=2,
            comment=im.info.get("comment", b""),
        )
    tmp_path.unlink(missing_ok=True)
    print(f"[OK] Saved optimized GIF to {output}")


def save_mp4_if_available(gif_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[INFO] ffmpeg not found; skipping MP4 generation.")
        return

    mp4_path = gif_path.with_suffix(".mp4")
    try:
        import imageio.v2 as imageio

        with imageio.get_reader(gif_path) as reader:
            frames = [f for f in reader]
            imageio.mimsave(mp4_path, frames, fps=8)
        print(f"[OK] Also saved {mp4_path}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[WARN] MP4 generation failed: {exc}")


def run_animation(data: DemoData, config: RenderConfig) -> None:
    levels = np.linspace(
        -max(data.global_w_max, 1e-6),
        max(data.global_w_max, 1e-6),
        52,
    )

    fig, ax_dash, ax_wig = configure_axes()
    fig.set_size_inches(config.figure_width, config.figure_height)
    fig.suptitle(
        "Real-time Calibration Simulation",
        fontsize=16,
        fontweight="bold",
        color=DARK,
        y=0.98,
    )
    fig.text(
        0.73,
        0.94,
        "TDM Optical Bus - squeezed-light calibration",
        ha="center",
        fontsize=11,
        color=GRAY,
    )

    def draw_frame(frame_idx: int) -> None:
        frame = data.frames[frame_idx]
        ax_dash.cla()
        ax_wig.cla()
        draw_dashboard(ax_dash, frame, frame_idx, config.n_phase1)
        draw_wigner_panel(ax_wig, frame, data.xvec, levels)

    print(f"Rendering {len(data.frames)} frames at {config.fps:.1f} fps ...")
    anim = FuncAnimation(fig, draw_frame, frames=len(data.frames), blit=False)
    tmp_path = config.output.with_suffix(".tmp.gif")
    anim.save(tmp_path, writer=PillowWriter(fps=config.fps), dpi=config.dpi)
    plt.close(fig)
    optimize_gif(tmp_path, config.output)

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
        save_mp4=args.save_mp4,
    )
    run_animation(data, render_cfg)


if __name__ == "__main__":
    main()
