"""Generate 2-column scenario dashboard PNG assets with consistent IEEE styling."""

from __future__ import annotations

import pathlib
import sys
from math import factorial

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Ensure package importability from scripts path.
SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import quantum_optical_bus.compat  # noqa: F401, E402

from quantum_optical_bus.hardware import WaveguideConfig, run_hardware_simulation  # noqa: E402
from quantum_optical_bus.interface import calculate_squeezing  # noqa: E402
from quantum_optical_bus.quantum import run_single_mode  # noqa: E402
from quantum_optical_bus.units import db_to_eta  # noqa: E402
from scripts.viz_style_ieee import (  # noqa: E402
    AXIS_COLOR,
    BG_COLOR,
    FIGURE_WIDTH_2COL_IN,
    SERIES_BLUE,
    SERIES_ORANGE,
    SERIES_PURPLE,
    SERIES_TEAL,
    apply_ieee_style,
    apply_layout,
    ieee_figsize,
    save_ieee,
    style_axis,
)

ASSETS_DIR = SRC_DIR.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

GRID_LIMIT = 4.0
GRID_POINTS = 120
X_VECTOR = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_POINTS)
CONTOUR_LEVELS = 24
FIG_SIZE = ieee_figsize(aspect=0.60)


def _run_quantum(r: float, theta: float, eta_loss: float) -> tuple[np.ndarray, float, float, float]:
    res = run_single_mode(r, theta, eta_loss, X_VECTOR)
    intrinsic_sq_db = -10.0 * np.log10(np.exp(-2.0 * r)) if r > 0 else 0.0
    return res.W, res.var_x, res.var_p, intrinsic_sq_db


def _draw_hardware(ax: plt.Axes, cfg: WaveguideConfig | None = None) -> None:
    if cfg is None:
        cfg = WaveguideConfig()
    n_eff, mode_area, ez_data, extent = run_hardware_simulation(cfg)
    ax.imshow(ez_data, extent=extent, cmap="RdBu", origin="lower", aspect="auto")
    ax.set_title("Waveguide mode profile", fontsize=11)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.text(
        0.04,
        0.96,
        f"n_eff = {n_eff:.2f}\nArea = {mode_area:.2f} μm²",
        transform=ax.transAxes,
        fontsize=8,
        color=AXIS_COLOR,
        va="top",
        ha="left",
    )
    style_axis(ax)


def _draw_calibration_curve(
    ax: plt.Axes,
    pump: float,
    *,
    title: str,
) -> tuple[float, float]:
    sq_powers = np.linspace(0.0, 500.0, 300)
    sq_db = -10.0 * np.log10(np.exp(-2.0 * calculate_squeezing(sq_powers))
    )
    r = calculate_squeezing(pump)
    observed = -10.0 * np.log10(np.exp(-2.0 * r)) if r > 0 else 0.0

    ax.plot(sq_powers, sq_db, color=SERIES_BLUE, lw=1.4)
    ax.axvline(pump, color=SERIES_ORANGE, ls="--", lw=1.1)
    ax.axhline(observed, color=SERIES_ORANGE, ls=":", lw=0.9, alpha=0.75)
    ax.scatter([pump], [observed], color=SERIES_ORANGE, s=36, zorder=5)
    style_axis(
        ax,
        title=title,
        xlabel="Pump power (mW)",
        ylabel="Intrinsic squeezing (dB)",
    )
    ax.set_xlim(0.0, 500.0)
    return observed, r


def _style_wigner_panel(
    ax: plt.Axes,
    frame: tuple[np.ndarray, float, float],
    title: str,
) -> None:
    image, r, intrinsic_sq_db = frame
    cf = ax.contourf(
        X_VECTOR,
        X_VECTOR,
        image,
        levels=CONTOUR_LEVELS,
        cmap="RdBu_r",
        vmin=-abs(image).max(),
        vmax=abs(image).max(),
    )
    ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    style_axis(
        ax,
        title=title,
        xlabel="x (SNU)",
        ylabel="p (SNU)",
    )
    ax.set_xlim(-GRID_LIMIT, GRID_LIMIT)
    ax.set_ylim(-GRID_LIMIT, GRID_LIMIT)
    ax.set_aspect("equal")


def scenario_vacuum() -> None:
    apply_ieee_style()
    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.30)

    W, var_x, var_p, _ = _run_quantum(0.0, 0.0, 1.0)

    ax_hw = fig.add_subplot(gs[:, 0])
    ax_cal = fig.add_subplot(gs[0, 1:])
    ax_wig = fig.add_subplot(gs[1, 1])
    ax_var = fig.add_subplot(gs[1, 2])

    _draw_hardware(ax_hw, WaveguideConfig())
    _draw_calibration_curve(ax_cal, 0.0, title="Calibration curve")
    _style_wigner_panel(ax_wig, (W, 0.0, 0.0), "Wigner function: vacuum")

    bars = ax_var.bar(
        ["Var(x)", "Var(p)", "Shot-noise"],
        [var_x, var_p, 0.5],
        color=[SERIES_BLUE, SERIES_ORANGE, AXIS_COLOR],
        edgecolor=BG_COLOR,
    )
    for bar, value in zip(bars, [var_x, var_p, 0.5]):
        ax_var.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=AXIS_COLOR,
        )
    style_axis(
        ax_var,
        title="Quadrature variance",
        xlabel="Metric",
        ylabel="Variance (SNU; vacuum=0.5)",
    )

    fig.suptitle("Scenario 1 — Vacuum baseline (P = 0 mW)", fontsize=11, y=0.985)
    fig.text(
        0.5,
        0.01,
        "Intrinsic squeezing is near 0 dB because pump power is zero.",
        ha="center",
        fontsize=9,
        color=AXIS_COLOR,
    )
    apply_layout(fig)
    out = ASSETS_DIR / "dashboard_vacuum.png"
    save_ieee(fig, out, dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")


def scenario_calibration() -> None:
    apply_ieee_style()
    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.30)

    pump = 200.0
    r = calculate_squeezing(pump)
    intrinsic_sq = -10.0 * np.log10(np.exp(-2.0 * r))
    W, _, _, _ = _run_quantum(pump, 0.0, 1.0)

    ax_hw = fig.add_subplot(gs[:, 0])
    ax_cal = fig.add_subplot(gs[0, 1:])
    ax_wig = fig.add_subplot(gs[1, 1])
    ax_pn = fig.add_subplot(gs[1, 2])

    _draw_hardware(ax_hw, WaveguideConfig())
    _draw_calibration_curve(ax_cal, pump, title="Calibration curve and operating point")
    _style_wigner_panel(ax_wig, (W, r, intrinsic_sq), f"Wigner function (r = {r:.3f})")

    max_n = 20
    ns = np.arange(max_n + 1)
    tanh_r = np.tanh(r)
    cosh_r = np.cosh(r)
    probs = np.zeros(max_n + 1)
    for n in range(0, max_n + 1, 2):
        k = n // 2
        probs[n] = (factorial(n) / (factorial(k) ** 2 * 4**k)) * (tanh_r**n) / cosh_r
    colors = [SERIES_BLUE if n % 2 == 0 else AXIS_COLOR for n in ns]
    ax_pn.bar(ns, probs, color=colors, edgecolor=BG_COLOR, width=0.75)
    style_axis(
        ax_pn,
        title="Photon number distribution P(n)",
        xlabel="Photon number n",
        ylabel="Probability",
    )
    ax_pn.set_xticks(ns)
    ax_pn.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Scenario 2 — Squeezed state (P = {pump:.0f} mW)",
        fontsize=11,
        y=0.985,
    )
    fig.text(
        0.5,
        0.01,
        f"Intrinsic squeezing at operating point: {intrinsic_sq:.2f} dB.",
        ha="center",
        fontsize=9,
        color=AXIS_COLOR,
    )
    apply_layout(fig)
    out = ASSETS_DIR / "dashboard_calibration.png"
    save_ieee(fig, out, dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")


def scenario_decoherence() -> None:
    apply_ieee_style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_2COL_IN, 4.7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    pump = 200.0
    r = calculate_squeezing(pump)
    loss_db_cm = 2.0
    length_mm = 5.0
    total_loss_db = loss_db_cm * (length_mm / 10.0)
    eta = float(db_to_eta(total_loss_db))

    W_pure, _, _, _ = _run_quantum(r, 0.0, 1.0)
    W_loss, _, _, obs_loss = _run_quantum(r, 0.0, eta)

    intrinsic_sq_db = -10.0 * np.log10(np.exp(-2.0 * r))
    ax_w_pure = fig.add_subplot(gs[0, 0])
    ax_w_loss = fig.add_subplot(gs[0, 1])
    ax_var = fig.add_subplot(gs[1, 0])
    ax_txt = fig.add_subplot(gs[1, 1])

    for axis, title, image in [
        (ax_w_pure, "Wigner: loss = 0 dB", W_pure),
        (ax_w_loss, f"Wigner: loss = {total_loss_db:.1f} dB", W_loss),
    ]:
        cf = axis.contourf(
            X_VECTOR,
            X_VECTOR,
            image,
            levels=CONTOUR_LEVELS,
            cmap="RdBu_r",
            vmin=-abs(image).max(),
            vmax=abs(image).max(),
        )
        axis.figure.colorbar(cf, ax=axis, fraction=0.046, pad=0.04)
        style_axis(axis, title=title, xlabel="x (SNU)", ylabel="p (SNU)")
        axis.set_aspect("equal")

    powers = np.linspace(0.0, 500.0, 200)
    var_curve = []
    for pw in powers:
        rr = calculate_squeezing(pw)
        vx = 0.5 * np.exp(-2.0 * rr)
        vx = eta * vx + (1 - eta) * 0.5
        var_curve.append(vx)

    ax_var.plot(
        powers,
        var_curve,
        color=SERIES_BLUE,
        lw=1.6,
        label="Var(x) after loss",
    )
    ax_var.axhline(0.5, color=AXIS_COLOR, ls="--", lw=1.0, label="Shot-noise limit")
    ax_var.axvline(pump, color=SERIES_ORANGE, ls="--", lw=1.0)
    ax_var.scatter([pump], [0.5 * np.exp(-2.0 * r)], color=SERIES_TEAL, s=35)
    style_axis(
        ax_var,
        title="Squeezing degradation from loss",
        xlabel="Pump power P (mW)",
        ylabel="Var(x) (SNU; vacuum=0.5)",
    )
    ax_var.set_yscale("log")
    ax_var.legend(loc="lower right", fontsize=7)

    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.text(
        0,
        1.0,
        "Decoherence summary",
        transform=ax_txt.transAxes,
        fontsize=11,
        fontweight="bold",
        color=SERIES_PURPLE,
    )
    ax_txt.text(0, 0.85, f"Pump power: {pump:.1f} mW", transform=ax_txt.transAxes, fontsize=9, color=AXIS_COLOR)
    ax_txt.text(
        0,
        0.77,
        f"Squeezing parameter: r = {r:.4f}",
        transform=ax_txt.transAxes,
        fontsize=9,
        color=AXIS_COLOR,
    )
    ax_txt.text(
        0,
        0.68,
        f"Total transmissivity: η = {eta:.4f}",
        transform=ax_txt.transAxes,
        fontsize=9,
        color=AXIS_COLOR,
    )
    ax_txt.text(
        0,
        0.60,
        f"Intrinsic (pre-loss): {intrinsic_sq_db:.2f} dB",
        transform=ax_txt.transAxes,
        fontsize=9,
        color=AXIS_COLOR,
    )
    ax_txt.text(
        0,
        0.52,
        f"Observed (post-loss): {obs_loss:.2f} dB",
        transform=ax_txt.transAxes,
        fontsize=9,
        color=SERIES_ORANGE,
    )
    ax_txt.text(
        0,
        0.14,
        "Loss decreases observed squeezing while intrinsic r stays constant.",
        transform=ax_txt.transAxes,
        fontsize=9,
        color=AXIS_COLOR,
    )

    fig.suptitle("Scenario 3 — Decoherence and channel loss", fontsize=11, y=0.985)
    fig.text(
        0.5,
        0.01,
        "Propagation loss restores the Wigner distribution toward the circular vacuum.",
        ha="center",
        fontsize=9,
        color=AXIS_COLOR,
    )
    apply_layout(fig)

    out = ASSETS_DIR / "dashboard_decoherence.png"
    save_ieee(fig, out, dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    print("Generating dashboard scenario gallery ...\n")
    scenario_vacuum()
    scenario_calibration()
    scenario_decoherence()
    print(f"\nDone. Images saved to: {ASSETS_DIR}")
