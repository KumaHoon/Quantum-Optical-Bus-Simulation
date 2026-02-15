"""
Generate advanced dashboard gallery images for CI artifacts.

Usage:
    python scripts/generate_advanced_dashboard_gallery.py

Outputs:
    assets/dashboard_multimode.png
    assets/dashboard_topology.png
    assets/dashboard_digital_twin.png
"""

import sys
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib.pyplot as plt

from quantum_optical_bus.multimode import run_multimode
from quantum_optical_bus.tdm_topology import simulate_topology
from quantum_optical_bus.estimation import fit_eta_and_loss
from quantum_optical_bus.control import simulate_phase_drift, apply_feedback_with_latency
from quantum_optical_bus.units import db_to_eta


ASSETS_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

BG = "#0d1117"
PANEL_BG = "#161b22"
WHITE = "#c9d1d9"
GRAY = "#8b949e"
BLUE = "#58a6ff"
RED = "#f97583"
GREEN = "#3fb950"


def _dark_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": WHITE,
            "text.color": WHITE,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "grid.color": "#30363d",
            "grid.alpha": 0.35,
            "font.size": 10,
        }
    )


def scenario_multimode():
    _dark_style()
    bins = 6
    xvec = np.linspace(-4.0, 4.0, 120)
    r = np.full(bins, 0.95)
    theta = np.linspace(0.0, 0.6, bins)
    loss_db = np.linspace(0.0, 3.0, bins)
    eta = db_to_eta(loss_db)
    res = run_multimode(r=r, theta=theta, eta_loss=eta, n_modes=bins, wigner_mode=2, xvec=xvec)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    ids = np.arange(bins)
    axs[0].plot(ids, res.observed_sq_db, marker="o", color=BLUE, label="Observed sq (dB)")
    axs[0].plot(ids, res.observed_antisq_db, marker="s", color=RED, label="Observed anti-sq (dB)")
    axs[0].set_xlabel("Bin")
    axs[0].set_ylabel("dB")
    axs[0].set_title("Per-bin squeezing metrics")
    axs[0].grid(True)
    axs[0].legend(loc="best", fontsize=8)

    axs[1].plot(ids, res.var_x, marker="o", color=GREEN, label="Var(x)")
    axs[1].plot(ids, res.var_p, marker="o", color=RED, label="Var(p)")
    axs[1].axhline(0.5, color=GRAY, ls="--", lw=1, label="Vacuum")
    axs[1].set_xlabel("Bin")
    axs[1].set_ylabel("Variance")
    axs[1].set_title("Per-bin variances")
    axs[1].grid(True)
    axs[1].legend(loc="best", fontsize=8)

    axs[2].contourf(xvec, xvec, res.wigner, levels=60, cmap="RdBu_r")
    axs[2].set_title("Selected-bin Wigner (bin=2)")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("p")
    axs[2].set_aspect("equal")

    fig.suptitle("Advanced Tab 1 - Multi-mode / Time-bin", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = ASSETS_DIR / "dashboard_multimode.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def scenario_topology():
    _dark_style()
    n = 6
    cfg = {
        "n_modes": n,
        "squeezing_r": [0.85] * n,
        "phase_shifts": (np.arange(n) * 0.15).tolist(),
        "loss": [1.0] * n,
        "couplings": [
            {"i": i, "j": i + 1, "theta": 0.35, "phi": 0.0, "eta_loss": float(db_to_eta(0.3))}
            for i in range(n - 1)
        ],
    }
    res = simulate_topology(cfg)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    im0 = axs[0].imshow(res.corr_x, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axs[0].set_title("Corr(X)")
    axs[0].set_xlabel("j")
    axs[0].set_ylabel("i")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(res.corr_p, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axs[1].set_title("Corr(P)")
    axs[1].set_xlabel("j")
    axs[1].set_ylabel("i")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    neighbors = np.arange(n - 1)
    axs[2].plot(neighbors, res.neighbor_cov_x, marker="o", color=BLUE, label="Cov(x_i,x_i+1)")
    axs[2].plot(neighbors, res.neighbor_cov_p, marker="o", color=RED, label="Cov(p_i,p_i+1)")
    axs[2].axhline(0.0, color=GRAY, ls="--", lw=1)
    axs[2].set_xlabel("Neighbor index")
    axs[2].set_ylabel("Covariance")
    axs[2].set_title("Neighbor correlations")
    axs[2].grid(True)
    axs[2].legend(loc="best", fontsize=8)

    fig.suptitle("Advanced Tab 2 - Topology Simulator", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = ASSETS_DIR / "dashboard_topology.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def scenario_digital_twin():
    _dark_style()
    rng = np.random.default_rng(17)
    eta_true = 0.115
    loss_true = 1.7
    powers = np.linspace(5.0, 180.0, 55)

    r_true = eta_true * np.sqrt(powers)
    trans_true = float(db_to_eta(loss_true))
    var_x_true = trans_true * (0.5 * np.exp(-2.0 * r_true)) + (1.0 - trans_true) * 0.5
    var_p_true = trans_true * (0.5 * np.exp(2.0 * r_true)) + (1.0 - trans_true) * 0.5

    data = {
        "timestamp": np.arange(powers.size, dtype=float),
        "pump_power_mw": powers,
        "measured_var_x": var_x_true + rng.normal(0.0, 0.003, size=powers.size),
        "measured_var_p": var_p_true + rng.normal(0.0, 0.009, size=powers.size),
        "estimated_loss_db": np.full(powers.size, 1.0),
    }
    eta_hat, loss_hat, diag = fit_eta_and_loss(data, model="variance")
    r_hat = eta_hat * np.sqrt(powers)
    trans_hat = float(db_to_eta(loss_hat))
    var_x_hat = trans_hat * (0.5 * np.exp(-2.0 * r_hat)) + (1.0 - trans_hat) * 0.5

    phase = simulate_phase_drift(T=260, step_sigma=0.015, drift_rate=0.0015, seed=21)
    latencies = np.arange(0, 9, dtype=int)
    rms = []
    retention = []
    for lat in latencies:
        ctl = apply_feedback_with_latency(
            latency_steps=int(lat),
            true_phase=phase,
            measurement_sigma=0.002,
            seed=23,
        )
        rms.append(ctl["rms_residual_phase_error"])
        retention.append(ctl["mean_retention_proxy"])

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axs[0].scatter(
        powers, data["measured_var_x"], s=16, alpha=0.8, color=BLUE, label="Measured Var(x)"
    )
    axs[0].plot(powers, var_x_hat, color=RED, lw=2, label="Fitted model Var(x)")
    axs[0].set_xlabel("Pump power (mW)")
    axs[0].set_ylabel("Variance")
    axs[0].set_title(
        f"Fit: eta={eta_hat:.4f} (true {eta_true:.4f}), "
        f"loss={loss_hat:.3f} dB (true {loss_true:.3f})"
    )
    axs[0].grid(True)
    axs[0].legend(loc="best", fontsize=8)

    axs[1].plot(latencies, rms, marker="o", color=RED, label="RMS residual phase")
    axs[1].plot(latencies, retention, marker="o", color=GREEN, label="Retention proxy")
    axs[1].set_xlabel("Latency steps")
    axs[1].set_title(f"Control trend (fit RMSE={diag['rmse']:.5f})")
    axs[1].grid(True)
    axs[1].legend(loc="best", fontsize=8)

    fig.suptitle("Advanced Tab 3 - Digital Twin Fit + Control", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = ASSETS_DIR / "dashboard_digital_twin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    print("Generating advanced dashboard gallery ...\n")
    scenario_multimode()
    scenario_topology()
    scenario_digital_twin()
    print("\nDone! Images saved to:", ASSETS_DIR)
