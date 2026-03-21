"""Generate IEEE-style advanced dashboard PNG artifacts for three workflows."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

from quantum_optical_bus.viz_style_ieee import (
    AXIS_COLOR,
    AXIS_LABEL_FONT_SIZE,
    SMALL_FONT_SIZE,
    compact_axis_formatter,
    SERIES_BLUE,
    SERIES_ORANGE,
    SERIES_TEAL,
    apply_review_layout,
    ieee_figsize,
    save_ieee,
    set_tab_title,
    set_review_axis,
)

try:
    from figstyle import (
        apply_style,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_meta,
    )
except ModuleNotFoundError:
    from scripts.figstyle import (
        apply_style,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_meta,
    )
try:
    from asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs
except ModuleNotFoundError:
    from scripts.asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs

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

FIG_SIZE = ieee_figsize(aspect=0.64)


def _apply_style(profile: str) -> None:
    apply_style(
        profile,
        base_font_size=11 if profile == "paper" else 11,
        tick_font_size=10 if profile == "paper" else 10,
        dpi=canonical_dpi(profile),
    )


def _save_profile_outputs(
    fig: plt.Figure,
    output_dir: pathlib.Path,
    name: str,
    profiles: tuple[str, ...],
    *,
    labels: dict[str, str],
    units: dict[str, str],
    notes: str,
    seed: int = 11,
    data_payload: dict[str, np.ndarray] | None = None,
) -> None:
    output_paths: list[pathlib.Path] = []
    for profile in profiles:
        output_paths.extend(resolve_outputs(output_dir / name, profile))
    for path in dict.fromkeys(output_paths):
        target_profile = path.parent.name
        target_dpi = canonical_dpi(target_profile)
        if path.parent.name == "paper":
            save_ieee(fig, path.with_suffix(".pdf"), dpi=300, skip_tight_layout=True)
            save_ieee(fig, path, dpi=target_dpi, skip_tight_layout=True)
        else:
            save_ieee(fig, path, dpi=target_dpi, skip_tight_layout=True)
        write_figure_meta(
            path,
            figure_id=path.stem,
            profile=target_profile,
            generator_script="scripts/generate_advanced_dashboard_gallery.py",
            generator_args=(f"--output-dir={output_dir}", f"--profile={target_profile}"),
            labels=labels,
            units=units,
            notes=notes,
            seed=seed,
            dpi=canonical_dpi(target_profile),
            canvas_px=canonical_canvas_px(target_profile),
            data_payload=data_payload,
        )
    print(f"[OK] {name}: {[str(path) for path in dict.fromkeys(output_paths)]}")


def _finish_layout(fig: plt.Figure, profile: str = "web") -> None:
    apply_review_layout(
        fig,
        mode=profile,
        left=0.08,
        right=0.88,
        bottom=0.11,
        top=0.90,
        wspace=0.46,
        hspace=0.36,
    )


def _set_top_label(fig: plt.Figure, text: str, profile: str = "web") -> None:
    set_tab_title(fig, text, mode=profile)


def _style_colormap(cbar, label: str) -> None:
    """Compact style for heatmap/Wigner colorbars."""
    cbar.set_label(label, fontsize=AXIS_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=AXIS_LABEL_FONT_SIZE - 0.5, pad=1)
    cbar.ax.yaxis.set_major_formatter(compact_axis_formatter())


def scenario_multimode(profile: str, output_dir: pathlib.Path) -> None:
    _apply_style(profile)
    xvec = np.linspace(-4.0, 4.0, 120)
    n = 6
    theta = np.linspace(0.0, 0.6, n)
    loss = np.linspace(0.0, 3.0, n)
    eta = units.db_to_eta(loss)

    result = run_multimode(
        r=np.full(n, 0.95),
        theta=theta,
        eta_loss=eta,
        n_modes=n,
        wigner_mode=2,
        xvec=xvec,
    )

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    ax_sq, ax_var, ax_wig = axes
    idx = np.arange(n)

    ax_sq.plot(
        idx,
        result.observed_sq_db,
        marker="o",
        color=SERIES_BLUE,
        label="Observed squeezing (dB)",
    )
    ax_sq.plot(
        idx,
        result.observed_antisq_db,
        marker="s",
        color=SERIES_ORANGE,
        label="Observed anti-sq (dB)",
    )
    set_review_axis(
        ax_sq,
        title="Per-bin squeezing",
        xlabel="Time-bin index",
        ylabel="Squeezing (dB)",
        integer_ticks=True,
        xticks=[float(v) for v in idx],
    )
    ax_sq.set_xticks(idx)
    ax_sq.legend(
        loc="upper right",
        bbox_to_anchor=(1.03, 1.0),
        borderaxespad=0.0,
        fontsize=SMALL_FONT_SIZE,
        frameon=False,
        ncol=1,
    )

    ax_var.plot(idx, result.var_x, marker="o", color=SERIES_TEAL, label="Var(x)")
    ax_var.plot(idx, result.var_p, marker="o", color=SERIES_ORANGE, label="Var(p)")
    ax_var.axhline(0.5, color=AXIS_COLOR, ls="--", lw=0.85, label="Vacuum (0.5)")
    set_review_axis(
        ax_var,
        title="Per-bin variances",
        xlabel="Time-bin index",
        ylabel="Variance (SNU; vacuum=0.5)",
        integer_ticks=True,
        xticks=[float(v) for v in idx],
    )
    ax_var.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax_var.set_xticks(idx)
    ax_var.legend(
        loc="upper right",
        bbox_to_anchor=(1.03, 1.0),
        borderaxespad=0.0,
        fontsize=SMALL_FONT_SIZE,
        frameon=False,
        ncol=1,
    )

    cset = ax_wig.contourf(xvec, xvec, result.wigner, levels=40, cmap="RdBu_r")
    cbar = ax_wig.figure.colorbar(cset, ax=ax_wig, fraction=0.042, pad=0.05)
    _style_colormap(cbar, "Wigner amplitude")
    set_review_axis(
        ax_wig,
        title="Selected Wigner (bin 2)",
        xlabel="x (SNU)",
        ylabel="p (SNU)",
        integer_ticks=True,
        xtick_format="%.2f",
        ytick_format="%.2f",
    )
    ax_wig.xaxis.set_major_formatter(compact_axis_formatter())
    ax_wig.yaxis.set_major_formatter(compact_axis_formatter())
    ax_wig.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax_wig.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax_wig.set_aspect("equal", adjustable="box")
    ax_wig.set_xticks(np.arange(-4, 5, 2))
    ax_wig.set_yticks(np.arange(-4, 5, 2))

    _set_top_label(fig, "Advanced Tab 1: Multi-mode / time-bin", profile)
    _finish_layout(fig, profile)
    _save_profile_outputs(
        fig,
        output_dir,
        "dashboard_multimode.png",
        (profile,),
        labels={
            "title": "Advanced dashboard: multimode",
            "xlabel": "Time-bin index",
            "ylabel": "Squeezing / variance / Wigner amplitude",
        },
        units={
            "time_bin_index": "index",
            "squeezing": "dB",
            "variance": "SNU",
            "wigner": "SNU",
        },
        notes="Multi-mode / time-bin scenario with observed squeezing and covariance-derived visual diagnostics.",
        data_payload={
            "time_bin_index": np.asarray(
                result.time_bins
                if hasattr(result, "time_bins")
                else np.arange(len(result.observed_sq_db))
            ),
            "observed_sq_db": np.asarray(result.observed_sq_db),
            "observed_antisq_db": np.asarray(result.observed_antisq_db),
            "var_x": np.asarray(result.var_x),
            "var_p": np.asarray(result.var_p),
        },
    )
    plt.close(fig)


def scenario_topology(profile: str, output_dir: pathlib.Path) -> None:
    _apply_style(profile)
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

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    ax_x, ax_p, ax_nei = axes
    im0 = ax_x.imshow(result.corr_x, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    cbar_x = ax_x.figure.colorbar(im0, ax=ax_x, fraction=0.042, pad=0.05)
    _style_colormap(cbar_x, "Corr(X)")
    set_review_axis(
        ax_x,
        title="Corr(X)",
        xlabel="Mode index",
        ylabel="Mode index",
        integer_ticks=True,
        xticks=[float(i) for i in range(n)],
        yticks=[float(i) for i in range(n)],
    )
    ax_x.set_xticks(np.arange(n))
    ax_x.set_yticks(np.arange(n))

    im1 = ax_p.imshow(result.corr_p, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    cbar_p = ax_p.figure.colorbar(im1, ax=ax_p, fraction=0.042, pad=0.05)
    _style_colormap(cbar_p, "Corr(P)")
    set_review_axis(
        ax_p,
        title="Corr(P)",
        xlabel="Mode index",
        ylabel="Mode index",
        integer_ticks=True,
        xticks=[float(i) for i in range(n)],
        yticks=[float(i) for i in range(n)],
    )
    ax_p.set_xticks(np.arange(n))
    ax_p.set_yticks(np.arange(n))

    neighbor = np.arange(n - 1)
    ax_nei.plot(
        neighbor,
        result.neighbor_cov_x,
        marker="o",
        color=SERIES_BLUE,
        label="Cov(X)",
    )
    ax_nei.plot(
        neighbor,
        result.neighbor_cov_p,
        marker="s",
        color=SERIES_ORANGE,
        label="Cov(P)",
    )
    ax_nei.axhline(0.0, color=AXIS_COLOR, ls="--", lw=0.85)
    set_review_axis(
        ax_nei,
        title="Neighbor covariance",
        xlabel="Neighbor pair index",
        ylabel="Covariance (SNU)",
        integer_ticks=True,
        xticks=[float(i) for i in neighbor],
    )
    ax_nei.yaxis.label.set_fontsize(8)
    ax_nei.set_xticks([float(i) for i in neighbor])
    ax_nei.grid(alpha=0.22)
    ax_nei.legend(
        loc="upper right",
        bbox_to_anchor=(1.03, 1.0),
        frameon=False,
        fontsize=SMALL_FONT_SIZE,
        ncol=1,
    )

    _set_top_label(fig, "Advanced Tab 2: Topology + BS couplings", profile)
    _finish_layout(fig, profile)
    _save_profile_outputs(
        fig,
        output_dir,
        "dashboard_topology.png",
        (profile,),
        labels={
            "title": "Advanced dashboard: topology",
            "xlabel": "Mode index",
            "ylabel": "Correlation / covariance",
        },
        units={
            "mode_index": "index",
            "neighbor_pair_index": "index",
            "corr": "correlation [-]",
            "cov": "SNU",
        },
        notes="Topology correlation blocks and neighbor covariance profile for coupled modes.",
        data_payload={
            "mode_index": np.arange(n, dtype=float),
            "corr_x": np.asarray(result.corr_x),
            "corr_p": np.asarray(result.corr_p),
            "neighbor_index": np.asarray(neighbor, dtype=float),
            "neighbor_cov_x": np.asarray(result.neighbor_cov_x),
            "neighbor_cov_p": np.asarray(result.neighbor_cov_p),
        },
    )
    plt.close(fig)


def scenario_digital_twin(profile: str, output_dir: pathlib.Path) -> None:
    _apply_style(profile)
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

    fig, (ax_fit, ax_ctrl) = plt.subplots(1, 2, figsize=FIG_SIZE)

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
        lw=1.4,
        label="Fitted Var(x)",
    )
    set_review_axis(
        ax_fit,
        title="Fit (synthetic benchmark)",
        xlabel="Pump power (mW)",
        ylabel="Variance (SNU; vacuum=0.5)",
        integer_ticks=False,
    )
    if eta_hat is not None:
        ax_fit.text(
            0.02,
            0.96,
            f"Fitted eta={eta_hat:.4f}, loss={loss_hat:.3f} dB",
            transform=ax_fit.transAxes,
            fontsize=SMALL_FONT_SIZE,
            color=AXIS_COLOR,
            ha="left",
            va="top",
        )
    ax_fit.legend(loc="upper right", fontsize=8, frameon=False)

    ax_ctrl_left = ax_ctrl
    ax_ctrl_right = ax_ctrl_left.twinx()
    ax_ctrl_left.plot(
        latencies,
        rms,
        marker="o",
        color=SERIES_ORANGE,
        label="RMS residual phase error",
    )
    ax_ctrl_left.set_ylabel(
        "RMS residual phase error (rad)",
        color=SERIES_ORANGE,
    )
    ax_ctrl_left.tick_params(axis="y", colors=SERIES_ORANGE)

    ax_ctrl_right.plot(
        latencies,
        retention,
        marker="s",
        color=SERIES_BLUE,
        label="Retention proxy",
    )
    ax_ctrl_right.set_ylabel("Retention proxy (unitless)", color=SERIES_BLUE)
    ax_ctrl_right.tick_params(axis="y", colors=SERIES_BLUE)
    set_review_axis(
        ax_ctrl_left,
        title="Latency sweep: control quality",
        xlabel="Latency steps",
        ylabel="RMS residual phase error (rad)",
        integer_ticks=True,
        xticks=[float(v) for v in latencies],
    )
    ax_ctrl_left.grid(alpha=0.25)

    lines_left, labels_left = ax_ctrl_left.get_legend_handles_labels()
    lines_right, labels_right = ax_ctrl_right.get_legend_handles_labels()
    ax_ctrl_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper left",
        fontsize=7,
    )
    ax_caption = fig.add_axes((0.06, 0.03, 0.92, 0.05), label="dt-caption")
    ax_caption.axis("off")
    ax_caption.text(
        0.0,
        0.0,
        "Synthetic benchmark: synthetic data used for fit and latency simulation.",
        fontsize=AXIS_LABEL_FONT_SIZE,
        color=AXIS_COLOR,
    )

    _set_top_label(fig, "Advanced Tab 3: Digital twin + latency control", profile)
    ax_caption.set_position([0.07, 0.03, 0.9, 0.06])
    _finish_layout(fig, profile)
    _save_profile_outputs(
        fig,
        output_dir,
        "dashboard_digital_twin.png",
        (profile,),
        labels={
            "title": "Advanced dashboard: digital twin",
            "xlabel": "Pump / latency",
            "ylabel": "Variance / residual / retention",
        },
        units={
            "pump_power_mw": "mW",
            "latency": "steps",
            "variance": "SNU",
            "residual": "rad",
            "retention": "unitless",
        },
        notes="Digital-twin workflow with synthetic fit and latency sweep retention diagnostics.",
        data_payload={
            "pump_power_mw": data["pump_power_mw"],
            "measured_var_x": data["measured_var_x"],
            "fitted_var_x": var_x_hat,
            "latency_steps": np.asarray(latencies, dtype=float),
            "rms_residual_phase_error": np.asarray(rms),
            "retention_proxy": np.asarray(retention),
        },
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ASSETS_DIR,
        help="Directory where advanced dashboard PNGs are written.",
    )
    parser.add_argument(
        "--profile",
        default="web",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    profiles = normalize_profiles(args.profile)

    print("Generating advanced dashboard gallery ...\n")
    for profile in profiles:
        scenario_multimode(profile, output_dir)
        scenario_topology(profile, output_dir)
        scenario_digital_twin(profile, output_dir)
    print(f"\nDone. Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
