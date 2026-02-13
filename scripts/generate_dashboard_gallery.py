"""
Dashboard Gallery Generator — produces presentation-ready images
that mirror the Calibration Dashboard's three key scenarios.

Usage:
    python scripts/generate_dashboard_gallery.py

Outputs to assets/:
    dashboard_vacuum.png       — Baseline (P = 0 mW)
    dashboard_calibration.png  — Calibration Curve + Squeezed State
    dashboard_decoherence.png  — Loss Channel / Decoherence demo
"""

import sys, pathlib

# Ensure package is importable
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import quantum_optical_bus.compat  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

import strawberryfields as sf
from strawberryfields.ops import Sgate, Rgate, LossChannel

from quantum_optical_bus.hardware import run_hardware_simulation, WaveguideConfig
from quantum_optical_bus.interface import calculate_squeezing

ASSETS_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Physics constants
ETA = 0.1
GRID_LIMIT = 4.0
GRID_POINTS = 120
xvec = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_POINTS)

# Dark aesthetic colours
BG        = "#0d1117"
PANEL_BG  = "#161b22"
ACCENT    = "#58a6ff"
RED       = "#f97583"
GREEN     = "#3fb950"
GRAY      = "#8b949e"
WHITE     = "#c9d1d9"


def _dark_style():
    """Apply a dark theme matching the Streamlit dashboard aesthetic."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": WHITE,
        "text.color": WHITE,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "grid.color": "#30363d",
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def _run_quantum(r, theta, eta_loss):
    """Single-mode Gaussian sim → Wigner, mean_photon, var_x, var_p, observed_sq_db."""
    prog = sf.Program(1)
    with prog.context as q:
        if r > 0:
            Sgate(r) | q[0]
        if theta != 0:
            Rgate(theta) | q[0]
        if eta_loss < 1.0:
            LossChannel(eta_loss) | q[0]
    result = sf.Engine("gaussian").run(prog)
    state = result.state
    W = state.wigner(0, xvec, xvec)
    mean_n = state.mean_photon(0)[0]
    cov = state.cov()
    var_x = cov[0, 0] / 2
    var_p = cov[1, 1] / 2
    # Observed squeezing from covariance eigenvalues
    V = cov / 2.0
    eigvals = np.linalg.eigvalsh(V)
    Vmin = eigvals[0]
    observed_sq_db = float(-10 * np.log10(Vmin / 0.5)) if Vmin > 0 else 0.0
    return W, mean_n, var_x, var_p, observed_sq_db


# ──────────────────────────────────────────────────────────────────
# Scenario 1 — VACUUM BASELINE (P = 0)
# ──────────────────────────────────────────────────────────────────
def scenario_vacuum():
    _dark_style()
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    pump = 0.0
    r = calculate_squeezing(pump)
    sq_db = 0.0
    W, _, var_x, var_p, _ = _run_quantum(r, 0.0, 1.0)

    # ---- Hardware mode ----
    cfg = WaveguideConfig()
    n_eff, mode_area, ez_data, extent = run_hardware_simulation(cfg)
    ax0 = fig.add_subplot(gs[:, 0])
    im = ax0.imshow(ez_data, extent=extent, cmap="RdBu", origin="lower", aspect="auto")
    ax0.set_title("Phase 1 · Waveguide Mode", color=ACCENT, fontsize=12, fontweight="bold")
    ax0.set_xlabel("x (μm)"); ax0.set_ylabel("y (μm)")
    ax0.text(0.05, 0.93, f"$n_{{eff}}$ = {n_eff:.2f}\nArea = {mode_area:.2f} μm²",
             transform=ax0.transAxes, fontsize=9,
             bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))

    # ---- Calibration (Top-Right) ----
    ax1 = fig.add_subplot(gs[0, 1:])
    powers_c = np.linspace(0, 500, 300)
    db_c = -10 * np.log10(np.exp(-2 * ETA * np.sqrt(powers_c)))
    ax1.plot(powers_c, db_c, color=ACCENT, lw=2)
    ax1.axvline(pump, color=RED, ls="--", lw=1.2)
    ax1.scatter([pump], [sq_db], color=RED, zorder=5, s=70)
    ax1.set_xlabel("Pump Power  P  (mW)")
    ax1.set_ylabel("Intrinsic Squeezing (pre-loss)  (dB)")
    ax1.set_title("Phase 2 · Calibration Curve  —  $r = \\eta\\sqrt{P}$",
                   color=ACCENT, fontsize=12, fontweight="bold")
    ax1.grid(True)
    ax1.text(260, 0.3, f"P = {pump:.0f} mW  →  r = {r:.3f}  →  Intrinsic: {sq_db:.1f} dB",
             fontsize=11, color=RED,
             bbox=dict(facecolor=PANEL_BG, edgecolor=RED, boxstyle="round,pad=0.4"))

    # ---- Wigner (Bottom-Right) ----
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.contourf(xvec, xvec, W, levels=60, cmap="RdBu_r")
    ax2.set_title("Wigner Function (Vacuum)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("x"); ax2.set_ylabel("p")
    ax2.set_aspect("equal")

    # ---- Noise Variance (Bottom-Right 2) ----
    ax3 = fig.add_subplot(gs[1, 2])
    cats = ["Var(x)", "Var(p)", "Shot Noise"]
    vals = [var_x, var_p, 0.5]
    colours = [GREEN, RED, GRAY]
    bars = ax3.bar(cats, vals, color=colours, edgecolor="#30363d", width=0.5)
    ax3.set_title("Quadrature Variance", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Variance")
    ax3.grid(axis="y")
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=9, color=WHITE)

    fig.suptitle("Scenario 1 — Baseline: Vacuum State  (P = 0 mW)",
                 fontsize=15, fontweight="bold", color=WHITE, y=0.98)
    fig.text(0.5, 0.01,
             "Insight: With zero pump power the bus transmits pure vacuum noise. "
             "The Wigner function is perfectly circular (isotropic).",
             ha="center", fontsize=10, color=GRAY)

    path = ASSETS_DIR / "dashboard_vacuum.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ──────────────────────────────────────────────────────────────────
# Scenario 2 — CALIBRATION CURVE (P = 200 mW, squeezed state)
# ──────────────────────────────────────────────────────────────────
def scenario_calibration():
    _dark_style()
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    pump = 200.0
    r = calculate_squeezing(pump)
    sq_db = -10 * np.log10(np.exp(-2 * r))
    W, mean_n, var_x, var_p, _ = _run_quantum(r, 0.0, 1.0)

    # ---- Hardware mode ----
    cfg = WaveguideConfig()
    n_eff, mode_area, ez_data, extent = run_hardware_simulation(cfg)
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(ez_data, extent=extent, cmap="RdBu", origin="lower", aspect="auto")
    ax0.set_title("Phase 1 · Waveguide Mode", color=ACCENT, fontsize=12, fontweight="bold")
    ax0.set_xlabel("x (μm)"); ax0.set_ylabel("y (μm)")
    ax0.text(0.05, 0.93, f"$n_{{eff}}$ = {n_eff:.2f}\nArea = {mode_area:.2f} μm²",
             transform=ax0.transAxes, fontsize=9,
             bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))

    # ---- Calibration Curve (Top-Right) ----
    ax1 = fig.add_subplot(gs[0, 1:])
    powers_c = np.linspace(0, 500, 300)
    db_c = -10 * np.log10(np.exp(-2 * ETA * np.sqrt(powers_c)))
    ax1.plot(powers_c, db_c, color=ACCENT, lw=2, label="$-10\\log_{10}(e^{-2r})$")
    ax1.axvline(pump, color=RED, ls="--", lw=1.2)
    ax1.axhline(sq_db, color=RED, ls=":", lw=0.8, alpha=0.6)
    ax1.scatter([pump], [sq_db], color=RED, zorder=5, s=70)
    ax1.set_xlabel("Pump Power  P  (mW)")
    ax1.set_ylabel("Intrinsic Squeezing (pre-loss)  (dB)")
    ax1.set_title("Phase 2 · Calibration Curve  —  $r = \\eta\\sqrt{P}$",
                   color=ACCENT, fontsize=12, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.text(pump + 15, sq_db + 0.15,
             f"P = {pump:.0f} mW\n$r$ = {r:.4f}\nIntrinsic: {sq_db:.2f} dB",
             fontsize=10, color=RED,
             bbox=dict(facecolor=PANEL_BG, edgecolor=RED, boxstyle="round,pad=0.4"))

    # ---- Wigner Function (Bottom-Left) ----
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.contourf(xvec, xvec, W, levels=60, cmap="RdBu_r")
    ax2.set_title(f"Wigner — Squeezed  (r = {r:.3f})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("x"); ax2.set_ylabel("p")
    ax2.set_aspect("equal")

    # ---- Photon Number Distribution (Bottom-Right) ----
    ax3 = fig.add_subplot(gs[1, 2])
    max_n = 20
    ns = np.arange(max_n + 1)
    tanh_r = np.tanh(r)
    cosh_r = np.cosh(r)
    probs = np.zeros(max_n + 1)
    from math import comb, factorial
    for n in range(0, max_n + 1, 2):
        k = n // 2
        probs[n] = (factorial(n) / (factorial(k)**2 * 4**k)) * (tanh_r**n) / cosh_r
    colours_pn = [ACCENT if n % 2 == 0 else "#30363d" for n in ns]
    ax3.bar(ns, probs, color=colours_pn, edgecolor="#30363d", width=0.7)
    ax3.set_xlabel("Photon number  $n$")
    ax3.set_ylabel("$P(n)$")
    ax3.set_title("Photon Number Distribution", fontsize=11, fontweight="bold")
    ax3.set_xticks(ns)
    ax3.grid(axis="y")

    fig.suptitle(f"Scenario 2 — Squeezed State  (P = {pump:.0f} mW,  r = {r:.4f})",
                 fontsize=15, fontweight="bold", color=WHITE, y=0.98)
    fig.text(0.5, 0.01,
             "Insight: The Wigner function is elliptical (amplitude-squeezed). "
             "Photon pairs appear exclusively in even-number states.",
             ha="center", fontsize=10, color=GRAY)

    path = ASSETS_DIR / "dashboard_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ──────────────────────────────────────────────────────────────────
# Scenario 3 — DECOHERENCE (P = 200 mW, loss = 2.0 dB/cm, L = 5 mm)
# ──────────────────────────────────────────────────────────────────
def scenario_decoherence():
    _dark_style()
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

    pump = 200.0
    r = calculate_squeezing(pump)
    loss_db_cm = 2.0
    wg_length_mm = 5.0
    total_loss_db = loss_db_cm * (wg_length_mm / 10.0)
    eta_loss = 10 ** (-total_loss_db / 10.0)

    # Pure squeezed state (no loss)
    W_pure, _, vx_pure, vp_pure, obs_sq_pure = _run_quantum(r, 0.0, 1.0)
    # Lossy state
    W_lossy, _, vx_lossy, vp_lossy, obs_sq_lossy = _run_quantum(r, 0.0, eta_loss)
    # Intrinsic squeezing (constant, pre-loss)
    intrinsic_sq_db = -10 * np.log10(np.exp(-2 * r))

    # ---- Wigner PURE (Top-Left) ----
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.contourf(xvec, xvec, W_pure, levels=60, cmap="RdBu_r")
    ax0.set_title("Pure Squeezed State\n(Loss = 0 dB)", fontsize=11, fontweight="bold", color=GREEN)
    ax0.set_xlabel("x"); ax0.set_ylabel("p")
    ax0.set_aspect("equal")

    # ---- Wigner LOSSY (Top-Right) ----
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax1.contourf(xvec, xvec, W_lossy, levels=60, cmap="RdBu_r")
    ax1.set_title(f"After Loss Channel\n(Loss = {total_loss_db:.1f} dB,  η = {eta_loss:.3f})",
                  fontsize=11, fontweight="bold", color=RED)
    ax1.set_xlabel("x"); ax1.set_ylabel("p")
    ax1.set_aspect("equal")

    # ---- Noise Variance Comparison (Bottom-Left) ----
    ax2 = fig.add_subplot(gs[1, 0:2])
    powers_nv = np.linspace(0, 500, 200)
    for eta_val, ls, label, col in [(1.0, "-", "No loss", GREEN),
                                     (eta_loss, "--", f"Loss = {total_loss_db:.1f} dB", RED)]:
        vx_arr = []
        for pw in powers_nv:
            rr = ETA * np.sqrt(pw)
            vx = 0.5 * np.exp(-2 * rr)
            vx = eta_val * vx + (1 - eta_val) * 0.5
            vx_arr.append(vx)
        ax2.plot(powers_nv, vx_arr, color=col, ls=ls, lw=2, label=f"Var(x) — {label}")
    ax2.axhline(0.5, color=GRAY, ls="--", lw=1, label="Shot noise")
    ax2.axvline(pump, color="#d2a8ff", ls="--", lw=1, alpha=0.7)
    ax2.set_xlabel("Pump Power  P  (mW)")
    ax2.set_ylabel("Var(x)")
    ax2.set_title("Squeezing Degradation from Loss", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True)

    # ---- Summary Metrics (Bottom-Right) ----
    ax3 = fig.add_subplot(gs[1, 2:4])
    ax3.axis("off")

    metrics_text = (
        f"━━━  Decoherence Summary  ━━━\n\n"
        f"Pump Power:          P = {pump:.0f} mW\n"
        f"Squeezing Parameter: r = {r:.4f}\n"
        f"Loss Rate:           {loss_db_cm:.1f} dB/cm\n"
        f"Waveguide Length:    {wg_length_mm:.1f} mm\n"
        f"Total Loss:          {total_loss_db:.1f} dB\n"
        f"Transmissivity:      η = {eta_loss:.4f}\n\n"
        f"━━━  Squeezing (dB)  ━━━\n\n"
        f"  Intrinsic (pre-loss):  {intrinsic_sq_db:.2f} dB\n"
        f"  Observed  (pure):      {obs_sq_pure:.2f} dB\n"
        f"  Observed  (lossy):     {obs_sq_lossy:.2f} dB\n\n"
        f"━━━  Quadrature Variances  ━━━\n\n"
        f"               Pure      Lossy\n"
        f"  Var(x):      {vx_pure:.4f}    {vx_lossy:.4f}\n"
        f"  Var(p):      {vp_pure:.4f}    {vp_lossy:.4f}\n"
        f"  Shot noise:  0.5000    0.5000\n\n"
        f"Loss reduces observed squeezing;\n"
        f"intrinsic r stays constant."
    )
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
             fontsize=10, color=WHITE, fontfamily="monospace", va="top",
             bbox=dict(facecolor=PANEL_BG, edgecolor=GRAY, boxstyle="round,pad=0.8"))

    fig.suptitle("Scenario 3 — Decoherence: Pure vs Lossy Squeezed State",
                 fontsize=15, fontweight="bold", color=WHITE, y=0.98)
    fig.text(0.5, 0.01,
             "Insight: Propagation loss degrades squeezing. The Wigner ellipse "
             "relaxes toward the vacuum circle — demonstrating decoherence in the CV quantum channel.",
             ha="center", fontsize=10, color=GRAY)

    path = ASSETS_DIR / "dashboard_decoherence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Calibration Dashboard gallery …\n")
    scenario_vacuum()
    scenario_calibration()
    scenario_decoherence()
    print("\nDone! Images saved to:", ASSETS_DIR)
