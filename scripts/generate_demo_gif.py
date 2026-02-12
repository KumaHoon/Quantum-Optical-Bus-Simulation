"""
Generate an animated GIF that demonstrates the Calibration Dashboard
power sweep: P = 0 → 400 mW, showing the Wigner function and
calibration metrics updating frame-by-frame.

Usage:
    python scripts/generate_demo_gif.py

Output:
    assets/demo_power_sweep.gif
"""

import sys, pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import quantum_optical_bus.compat  # noqa: F401

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

import strawberryfields as sf
from strawberryfields.ops import Sgate

from quantum_optical_bus.interface import calculate_squeezing

ASSETS_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
ETA = 0.1
GRID_LIMIT = 4.0
GRID_POINTS = 80
xvec = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_POINTS)
POWERS = np.concatenate([
    np.linspace(0, 400, 30),   # ramp up
    np.linspace(400, 0, 30),   # ramp down
])

# Dark theme
BG = "#0d1117"
PANEL = "#161b22"
ACCENT = "#58a6ff"
RED = "#f97583"
GREEN = "#3fb950"
GRAY = "#8b949e"
WHITE = "#c9d1d9"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": WHITE,
    "text.color": WHITE,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "grid.color": "#30363d",
    "grid.alpha": 0.4,
    "font.size": 10,
})

# Pre-compute Wigner functions for each power level
print("Pre-computing Wigner functions ...")
wigner_cache = {}
for pw in np.unique(POWERS.round(1)):
    r = calculate_squeezing(pw)
    prog = sf.Program(1)
    with prog.context as q:
        if r > 0:
            Sgate(r) | q[0]
    result = sf.Engine("gaussian").run(prog)
    W = result.state.wigner(0, xvec, xvec)
    cov = result.state.cov()
    wigner_cache[round(pw, 1)] = (W, r, cov[0, 0]/2, cov[1, 1]/2)

# Pre-compute calibration curve
powers_curve = np.linspace(0, 500, 300)
db_curve = -10 * np.log10(np.exp(-2 * ETA * np.sqrt(powers_curve)))

# --- Figure setup ---
fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                       width_ratios=[1.2, 1.0, 0.6])

# Calibration curve axis
ax_cal = fig.add_subplot(gs[0, 0])
ax_cal.set_xlabel("Pump Power  P  (mW)")
ax_cal.set_ylabel("Squeezing  (dB)")
ax_cal.set_title("Calibration Curve", color=ACCENT, fontsize=12, fontweight="bold")
ax_cal.set_xlim(0, 500)
ax_cal.set_ylim(0, 21)
ax_cal.grid(True)
cal_line, = ax_cal.plot(powers_curve, db_curve, color=ACCENT, lw=2)
vline = ax_cal.axvline(0, color=RED, ls="--", lw=1.2)
dot, = ax_cal.plot([], [], "o", color=RED, markersize=8, zorder=5)
cal_text = ax_cal.text(0.55, 0.35, "", transform=ax_cal.transAxes, fontsize=10,
                       color=RED, bbox=dict(facecolor=PANEL, edgecolor=RED,
                                            boxstyle="round,pad=0.4"))

# Wigner axis
ax_wig = fig.add_subplot(gs[0, 1])
ax_wig.set_xlabel("x")
ax_wig.set_ylabel("p")
ax_wig.set_aspect("equal")

# Metrics axis
ax_met = fig.add_subplot(gs[0, 2])
ax_met.axis("off")

fig.suptitle("TDM Optical Bus -- Calibration Dashboard Demo",
             fontsize=14, fontweight="bold", color=WHITE, y=0.97)


def update(frame):
    pw = round(POWERS[frame], 1)
    W, r, vx, vp = wigner_cache[pw]
    sq_db = -10 * np.log10(np.exp(-2 * r)) if r > 0 else 0.0

    # Calibration curve
    vline.set_xdata([pw, pw])
    dot.set_data([pw], [sq_db])
    cal_text.set_text(f"P = {pw:.0f} mW\nr = {r:.3f}\n{sq_db:.1f} dB")

    # Wigner
    ax_wig.cla()
    ax_wig.contourf(xvec, xvec, W, levels=50, cmap="RdBu_r")
    ax_wig.set_xlabel("x")
    ax_wig.set_ylabel("p")
    ax_wig.set_title(f"Wigner Function  (r = {r:.3f})",
                     fontsize=11, fontweight="bold", color=WHITE)
    ax_wig.set_aspect("equal")

    # Metrics
    ax_met.cla()
    ax_met.axis("off")
    metrics = (
        f"Power\n{pw:.0f} mW\n\n"
        f"r\n{r:.4f}\n\n"
        f"dB\n{sq_db:.2f}\n\n"
        f"Var(x)\n{vx:.4f}\n\n"
        f"Var(p)\n{vp:.4f}"
    )
    ax_met.text(0.5, 0.95, metrics, transform=ax_met.transAxes,
                fontsize=11, color=WHITE, ha="center", va="top",
                fontfamily="monospace",
                bbox=dict(facecolor=PANEL, edgecolor=GRAY,
                          boxstyle="round,pad=0.6"))

    return [vline, dot, cal_text]


print(f"Rendering {len(POWERS)} frames ...")
anim = FuncAnimation(fig, update, frames=len(POWERS), blit=False)

output_path = ASSETS_DIR / "demo_power_sweep.gif"
anim.save(str(output_path), writer=PillowWriter(fps=6), dpi=100)
plt.close(fig)
print(f"[OK] Saved to {output_path}")
