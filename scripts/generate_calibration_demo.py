"""
Generate a professional calibration demo GIF for grad school portfolio.

Two-phase animation:
  Phase 1 -- Power ramp (0 -> 200 mW): vacuum circle flattens into squeezed ellipse
  Phase 2 -- Loss ramp  (0 -> 2 dB):   squeezed ellipse becomes round/fuzzy (decoherence)

Output: assets/calibration_demo.gif

Usage:
    python scripts/generate_calibration_demo.py
"""

import sys
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import quantum_optical_bus.compat  # noqa: F401

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches

import strawberryfields as sf
from strawberryfields.ops import Sgate, LossChannel

from quantum_optical_bus.interface import calculate_squeezing
from quantum_optical_bus.units import db_to_eta, observed_squeezing_from_cov, sf_cov_to_vacuum05

# -- Output ----------------------------------------------------------------
ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
OUT_PATH = ASSETS / "calibration_demo.gif"

# -- Physics constants -----------------------------------------------------
GRID_LIM = 5.0
GRID_PTS = 120  # higher resolution for smoother contours
xvec = np.linspace(-GRID_LIM, GRID_LIM, GRID_PTS)

# -- Animation timeline ----------------------------------------------------
N_PHASE1 = 40  # frames: power 0 -> 200 mW
N_PHASE2 = 30  # frames: loss  0 -> 2 dB
N_TOTAL = N_PHASE1 + N_PHASE2

powers = np.concatenate(
    [
        np.linspace(0, 200, N_PHASE1),
        np.full(N_PHASE2, 200.0),
    ]
)
losses_db = np.concatenate(
    [
        np.zeros(N_PHASE1),
        np.linspace(0, 2.0, N_PHASE2),
    ]
)

# -- Pre-compute all Wigner functions --------------------------------------
print(f"Pre-computing {N_TOTAL} Wigner frames ...")
cache = []
for i in range(N_TOTAL):
    P = powers[i]
    L_db = losses_db[i]

    r = calculate_squeezing(P)
    intrinsic_sq_db = -10 * np.log10(np.exp(-2 * r)) if r > 0 else 0.0
    transmittance = float(db_to_eta(L_db))

    prog = sf.Program(1)
    with prog.context as q:
        if r > 0:
            Sgate(r) | q[0]
        if transmittance < 1.0:
            LossChannel(transmittance) | q[0]

    result = sf.Engine("gaussian").run(prog)
    W = result.state.wigner(0, xvec, xvec)
    cov = sf_cov_to_vacuum05(result.state.cov())
    var_x = float(cov[0, 0])
    var_p = float(cov[1, 1])
    observed_sq_db, _ = observed_squeezing_from_cov(cov)

    cache.append(
        {
            "W": W,
            "r": r,
            "intrinsic_sq_db": intrinsic_sq_db,
            "observed_sq_db": observed_sq_db,
            "P": P,
            "L_db": L_db,
            "transmittance": transmittance,
            "var_x": var_x,
            "var_p": var_p,
        }
    )
print("Done.")

# -- Whitesmoke lab theme --------------------------------------------------
BG = "#f5f5f5"  # whitesmoke
PANEL = "#ffffff"  # white panels
ACCENT = "#1a73e8"  # blue
RED = "#d93025"  # bold red
GREEN = "#188038"  # green
ORANGE = "#e37400"  # orange
GRAY = "#5f6368"  # text gray
DARK = "#202124"  # dark text
LIGHT_BRD = "#dadce0"  # light border

# Vacuum reference circle (radius = sqrt(0.5) for shot noise in x-p space)
VACUUM_RADIUS = np.sqrt(1.0)  # unit circle for reference

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
        "font.size": 13,
    }
)

# -- Figure layout ---------------------------------------------------------
fig = plt.figure(figsize=(15, 7))

# Left panel: dashboard metrics
ax_dash = fig.add_axes([0.03, 0.06, 0.38, 0.82])
ax_dash.set_xlim(0, 10)
ax_dash.set_ylim(0, 10)
ax_dash.axis("off")

# Right panel: Wigner function
ax_wig = fig.add_axes([0.48, 0.06, 0.48, 0.82])

# Title
fig.text(
    0.50,
    0.95,
    "Figure 1: Real-time Calibration Simulation",
    ha="center",
    fontsize=18,
    fontweight="bold",
    color=DARK,
)
fig.text(
    0.50,
    0.91,
    "TDM Optical Bus  \u2014  Squeezed Light Source",
    ha="center",
    fontsize=13,
    color=GRAY,
)


def draw_bar(ax, x, y, w, h, frac, color, label_left, label_right):
    """Draw a horizontal progress bar."""
    # Background track
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor="#e8eaed",
            edgecolor=LIGHT_BRD,
            linewidth=0.8,
        )
    )
    # Filled portion
    if frac > 0.005:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                max(w * frac, 0.15),
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
        fontsize=14,
        color=GRAY,
        fontweight="bold",
    )
    ax.text(
        x + w + 0.15,
        y + h / 2,
        label_right,
        ha="left",
        va="center",
        fontsize=15,
        color=DARK,
        fontweight="bold",
    )


def update(frame):
    d = cache[frame]
    phase = "Phase 1: Power Sweep" if frame < N_PHASE1 else "Phase 2: Loss Channel"
    phase_color = ACCENT if frame < N_PHASE1 else ORANGE

    # -- Left panel --
    ax_dash.cla()
    ax_dash.set_xlim(0, 10)
    ax_dash.set_ylim(0, 10)
    ax_dash.axis("off")

    # Phase indicator pill
    ax_dash.add_patch(
        mpatches.FancyBboxPatch(
            (0.3, 8.7),
            9.2,
            1.0,
            boxstyle="round,pad=0.18",
            facecolor=phase_color,
            edgecolor="none",
            alpha=0.12,
        )
    )
    ax_dash.text(
        4.9, 9.2, phase, ha="center", va="center", fontsize=15, fontweight="bold", color=phase_color
    )

    # Pump Power bar
    draw_bar(ax_dash, 3.2, 7.4, 5.2, 0.6, d["P"] / 200.0, ACCENT, "Pump Power", f"{d['P']:.0f} mW")

    # Loss bar
    draw_bar(
        ax_dash, 3.2, 6.3, 5.2, 0.6, d["L_db"] / 2.0, ORANGE, "Prop. Loss", f"{d['L_db']:.2f} dB"
    )

    # Divider
    ax_dash.plot([0.5, 9.5], [5.85, 5.85], color=LIGHT_BRD, lw=1.0)

    # Section header
    ax_dash.text(
        4.9,
        5.35,
        "CALIBRATION METRICS",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=GRAY,
        style="italic",
    )

    # r parameter
    ax_dash.text(0.8, 4.4, "Squeezing Param (r):", fontsize=14, color=GRAY)
    ax_dash.text(9.0, 4.4, f"{d['r']:.4f}", fontsize=16, fontweight="bold", color=GREEN, ha="right")

    # ---- Squeezing dB: show BOTH intrinsic and observed ----
    ax_dash.add_patch(
        mpatches.FancyBboxPatch(
            (0.4, 2.2),
            9.2,
            2.2,
            boxstyle="round,pad=0.18",
            facecolor="#fce8e6",
            edgecolor=RED,
            linewidth=2.0,
        )
    )
    ax_dash.text(
        4.9,
        4.0,
        "INTRINSIC SQUEEZING (pre-loss)",
        ha="center",
        fontsize=10,
        color=GRAY,
        fontweight="bold",
    )
    ax_dash.text(
        4.9,
        3.4,
        f"{d['intrinsic_sq_db']:.2f} dB",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color=GRAY,
    )
    ax_dash.text(
        4.9,
        2.85,
        "OBSERVED SQUEEZING (post-loss)",
        ha="center",
        fontsize=10,
        color=RED,
        fontweight="bold",
    )
    ax_dash.text(
        4.9,
        2.35,
        f"{d['observed_sq_db']:.2f} dB",
        ha="center",
        fontsize=24,
        fontweight="bold",
        color=RED,
    )

    # Transmissivity
    ax_dash.text(0.8, 1.0, "Transmissivity:", fontsize=13, color=GRAY)
    eta_str = f"{d['transmittance']:.4f}"
    if d["transmittance"] > 0.9:
        eta_color = GREEN
    elif d["transmittance"] > 0.5:
        eta_color = ORANGE
    else:
        eta_color = RED
    ax_dash.text(5.5, 1.0, eta_str, fontsize=14, fontweight="bold", color=eta_color)

    # Frame counter
    ax_dash.text(
        4.9, 0.15, f"Frame {frame + 1}/{N_TOTAL}", ha="center", fontsize=10, color=GRAY, alpha=0.5
    )

    # -- Right panel: Wigner --
    ax_wig.cla()
    # Smooth contour with many levels
    ax_wig.contourf(xvec, xvec, d["W"], levels=120, cmap="RdBu_r")
    ax_wig.contour(xvec, xvec, d["W"], levels=30, colors="k", linewidths=0.15, alpha=0.3)

    # Vacuum reference circle (dashed)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_wig.plot(
        VACUUM_RADIUS * np.cos(theta),
        VACUUM_RADIUS * np.sin(theta),
        linestyle="--",
        color="#555555",
        linewidth=1.2,
        alpha=0.7,
        label="Vacuum reference",
    )
    ax_wig.legend(loc="upper right", fontsize=10, framealpha=0.8, edgecolor=LIGHT_BRD)

    ax_wig.set_xlim(-GRID_LIM, GRID_LIM)
    ax_wig.set_ylim(-GRID_LIM, GRID_LIM)
    ax_wig.set_xlabel("x  (position quadrature)", fontsize=14)
    ax_wig.set_ylabel("p  (momentum quadrature)", fontsize=14)
    ax_wig.set_title(
        f"Wigner Function    r = {d['r']:.3f}", fontsize=15, fontweight="bold", color=DARK, pad=10
    )
    ax_wig.set_aspect("equal")
    ax_wig.grid(True, alpha=0.12, color=GRAY)


print(f"Rendering {N_TOTAL} frames ...")
anim = FuncAnimation(fig, update, frames=N_TOTAL, blit=False)
anim.save(str(OUT_PATH), writer=PillowWriter(fps=8), dpi=120)
plt.close(fig)
print(f"[OK] Saved to {OUT_PATH}")
