"""Generate IEEE-style 2-column advanced dashboard PNG artifacts."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from quantum_optical_bus.viz_style import (
    AXIS_COLOR,
    BASE_FONT_SIZE,
    FIGURE_WIDTH_2COL_IN,
    SERIES_BLUE,
    SERIES_ORANGE,
    SERIES_TEAL,
    apply_ieee_style,
    apply_layout,
    style_axis,
)

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import quantum_optical_bus.compat  # noqa: F401, E402

import quantum_optical_bus.units as units  # noqa: E402
from quantum_optical_bus.multimode import run_multimode  # noqa: E402
from quantum_optical_bus.tdm_topology import simulate_topology  # noqa: E402
from quantum_optical_bus.estimation import fit_eta_and_loss  # noqa: E402
from quantum_optical_bus.control import simulate_phase_drift, apply_feedback_with_latency  # noqa: E402

ASSETS_DIR = SRC_DIR.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _apply_plot_defaults() -> None:
    apply_ieee_style(base_font_size=BASE_FONT_SIZE, tick_font_size=9)


def scenario_multimode() -> None:
    _apply_plot_defaults()

    xvec = np.linspace(-4.0, 4.0, 120)
    n = 6
    r = np.full(n, 0.95)
    theta = np.linspace(0.0, 0.6, n)
    loss = np.linspace(0.0, 3.0, n)
    eta = units.db_to_eta(loss)

    result = run_multimode(
        r=r,
        theta=theta,
        eta_loss=eta,
        n_modes=n,
        wigner_mode=2,
        xvec=xvec,
    )

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_2COL_IN, 4.0))
    ax_sq, ax_var, ax_wig = axes

    idx = np.arange(n)
    ax_sq.plot(idx, result.observed_sq_db, marker="o", color=SERIES_BLUE, label="Observed sq (dB)")
    ax_sq.plot(
        idx,
        result.observed_antisq_db,
        marker="s",
        color=SERIES_ORANGE,
        label="Observed anti-sq (dB)",
    )
    style_axis(
        ax_sq,
        title="Per-bin squeezing metrics",
        xlabel="Time-bin index",
        ylabel="Squeezing (dB)",
    )
    ax_sq.legend(loc="best", fontsize=8)

    ax_var.plot(idx, result.var_x, marker="o", color=SERIES_TEAL, label="Var(x)")
    ax_var.plot(idx, result.var_p, marker="o", color=SERIES_ORANGE, label="Var(p)")
    ax_var.axhline(0.5, color=AXIS_COLOR, ls="--", lw=1.0, label="Shot-noise limit")
    style_axis(
        ax_var,
        title="Per-bin quadrature variances",
        xlabel="Time-bin index",
        ylabel="Variance (SNU; vacuum=0.5)",
    )
    ax_var.legend(loc="best", fontsize=8)

    cset = ax_wig.contourf(xvec, xvec, result.wigner, levels=40, cmap="RdBu_r")
    fig.colorbar(cset, ax=ax_wig, fraction=0.046, pad=0.04)
    style_axis(
        ax_wig,
        title="Selected-bin Wigner (bin=2)",
        xlabel="x",
        ylabel="p",
    )
    ax_wig.set_aspect("equal")

    fig.suptitle(
        "Advanced Tab 1 - Multi-mode / time-bin simulation",
        fontsize=12,
        y=0.98,
    )
    apply_layout(fig)
    out = ASSETS_DIR / "dashboard_multimode.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def scenario_topology() -> None:
    _apply_plot_defaults()
    n = 6
    cfg = {
        "n_modes": n,
        "squeezing_r": [0.85] * n,
        "phase_shifts": (np.arange(n) * 0.15).tolist(),
        "loss": [1.0] * n,
        "couplings": [
            {
                "i": i,
                "j": i + 1,
                "theta": 0.35,
                "phi": 0.0,
                "eta_loss": float(units.db_to_eta(0.3)),
            }
            for i in range(n - 1)
        ],
    }
    result = simulate_topology(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_2COL_IN, 4.0))
    ax_x, ax_p, ax_nei = axes

    im0 = ax_x.imshow(result.corr_x, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax_x.set_title("Corr(X)")
    ax_x.set_xlabel("Time-bin index")
    ax_x.set_ylabel("Time-bin index")
    fig.colorbar(im0, ax=ax_x, fraction=0.046, pad=0.04)

    im1 = ax_p.imshow(result.corr_p, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax_p.set_title("Corr(P)")
    ax_p.set_xlabel("Time-bin index")
    ax_p.set_ylabel("Time-bin index")
    fig.colorbar(im1, ax=ax_p, fraction=0.046, pad=0.04)

    neighbor = np.arange(n - 1)
    ax_nei.plot(
        neighbor,
        result.neighbor_cov_x,
        marker="o",
        color=SERIES_BLUE,
        label="Cov(X_i, X_{i+1})",
    )
    ax_nei.plot(
        neighbor,
        result.neighbor_cov_p,
        marker="o",
        color=SERIES_ORANGE,
        label="Cov(P_i, P_{i+1})",
    )
    ax_nei.axhline(0.0, color=AXIS_COLOR, ls="--", lw=1.0)
    style_axis(
        ax_nei,
        title="Neighbor correlations",
        xlabel="Neighbor pair index",
        ylabel="Covariance (SNU)",
    )
    ax_nei.legend(loc="best", fontsize=8)

    fig.suptitle("Advanced Tab 2 - Topology simulator", fontsize=12, y=0.98)
    apply_layout(fig)
    out = ASSETS_DIR / "dashboard_topology.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def scenario_digital_twin() -> None:
    _apply_plot_defaults()
    rng = np.random.default_rng(17)
    eta_true = 0.115
    loss_true = 1.7
    powers = np.linspace(5.0, 180.0, 55)

    r_true = eta_true * np.sqrt(powers)
    trans_true = float(units.db_to_eta(loss_true))
    var_x_true = trans_true * (0.5 * np.exp(-2.0 * r_true)) + (1.0 - trans_true) * 0.5
    var_p_true = trans_true * (0.5 * np.exp(2.0 * r_true)) + (1.0 - trans_true) * 0.5

    data = {
        "timestamp": np.arange(powers.size, dtype=float),
        "pump_power_mw": powers,
        "measured_var_x": var_x_true + rng.normal(0.0, 0.003, size=powers.size),
        "measured_var_p": var_p_true + rng.normal(0.0, 0.009, size=powers.size),
        "estimated_loss_db": np.full(powers.size, 1.0),
    }
    eta_hat, loss_hat, _diag = fit_eta_and_loss(data, model="variance")
    r_hat = eta_hat * np.sqrt(powers)
    trans_hat = float(units.db_to_eta(loss_hat))
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

    fig, (ax_fit, ax_ctrl) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_2COL_IN, 4.0))

    ax_fit.scatter(
        powers,
        data["measured_var_x"],
        s=16,
        alpha=0.8,
        color=SERIES_BLUE,
        label="Measured Var(x)",
    )
    ax_fit.plot(
        powers,
        var_x_hat,
        color=SERIES_ORANGE,
        lw=2,
        label="Fitted model Var(x)",
    )
    style_axis(
        ax_fit,
        title=(
            f"Fit: eta={eta_hat:.4f} (true {eta_true:.4f}), "
            f"loss={loss_hat:.3f} dB (true {loss_true:.3f})"
        ),
        xlabel="Pump power (mW)",
        ylabel="Variance (SNU; vacuum=0.5)",
    )
    ax_fit.legend(loc="best", fontsize=8)

    ax_rms = ax_ctrl.twinx()
    ax_ctrl.plot(
        latencies,
        rms,
        marker="o",
        color=SERIES_ORANGE,
        label="RMS residual phase error",
    )
    ax_ctrl.set_ylabel("RMS residual phase error (rad)")
    ax_ctrl.tick_params(axis="y", colors=SERIES_ORANGE)
    ax_ctrl.set_xlabel("Latency steps")

    ax_rms.plot(
        latencies,
        retention,
        marker="s",
        color=SERIES_BLUE,
        label="Retention proxy",
    )
    ax_rms.set_ylabel("Retention proxy (unitless)")
    ax_rms.tick_params(axis="y", colors=SERIES_BLUE)

    lines_left, labels_left = ax_ctrl.get_legend_handles_labels()
    lines_right, labels_right = ax_rms.get_legend_handles_labels()
    ax_ctrl.legend(lines_left + lines_right, labels_left + labels_right, loc="best", fontsize=8)
    style_axis(ax_ctrl, title="Control trend", xlabel="Latency steps")
    ax_ctrl.grid(alpha=0.25)

    fig.suptitle("Advanced Tab 3 - Digital twin fit + control", fontsize=12, y=0.98)
    apply_layout(fig)
    out = ASSETS_DIR / "dashboard_digital_twin.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    print("Generating advanced dashboard gallery ...\n")
    scenario_multimode()
    scenario_topology()
    scenario_digital_twin()
    print(f"\nDone. Images saved to: {ASSETS_DIR}")
