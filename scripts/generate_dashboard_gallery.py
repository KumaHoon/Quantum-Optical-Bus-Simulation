"""Generate 2-column scenario dashboard PNG assets with consistent multi-profile styling."""

from __future__ import annotations

import argparse
import pathlib
import sys
from math import factorial
import textwrap
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

# Ensure package importability from scripts path.
SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import quantum_optical_bus.compat  # noqa: F401, E402

from quantum_optical_bus.hardware import WaveguideConfig, run_hardware_simulation  # noqa: E402
from quantum_optical_bus.interface import calculate_squeezing  # noqa: E402
from quantum_optical_bus.quantum import run_single_mode  # noqa: E402
from quantum_optical_bus.units import db_to_eta  # noqa: E402
from quantum_optical_bus.viz_style_ieee import (  # noqa: E402
    AXIS_COLOR,
    BG_COLOR,
    AXIS_LABEL_FONT_SIZE,
    AXIS_TITLE_FONT_SIZE,
    compact_axis_formatter,
    SERIES_BLUE,
    SERIES_ORANGE,
    apply_review_layout,
    SERIES_PURPLE,
    SERIES_TEAL,
    set_review_axis,
    set_tab_title,
    save_ieee,
    style_axis,
)

try:
    from figstyle import (
        apply_style,
        canonical_canvas_inches,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_meta,
    )
except ModuleNotFoundError:
    from scripts.figstyle import (
        apply_style,
        canonical_canvas_inches,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_meta,
    )
try:
    from asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs
except ModuleNotFoundError:
    from scripts.asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs

ASSETS_DIR = SRC_DIR.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

GRID_LIMIT = 4.0
GRID_POINTS = 120
X_VECTOR = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_POINTS)
CONTOUR_LEVELS = 24
FIG_HSPACE = 0.46
FIG_WSPACE = 0.36
WIGNER_OFFSET_X_PX = 0.0
WIGNER_OFFSET_Y_PX = -35.0
SUMMARY_TITLE_FONT_SIZE = 12
SUMMARY_LABEL_FONT_SIZE = 8
SUMMARY_VALUE_FONT_SIZE = 8
SUMMARY_NOTE_FONT_SIZE = 7
SUMMARY_LABEL_X = 0.04
SUMMARY_VALUE_X = 0.62
SUMMARY_LINE_GAP = 0.064
SUMMARY_NOTE_GAP = 0.05
FOOTNOTE_X = 0.04
FOOTNOTE_Y_WEB = 0.08
FOOTNOTE_Y_PAPER = 0.07
FOOTNOTE_FONT_SIZE = 7


def _apply_style(profile: str) -> None:
    if profile == "paper":
        apply_style(profile, base_font_size=10, tick_font_size=9, dpi=canonical_dpi(profile))
    else:
        apply_style(profile, base_font_size=9, tick_font_size=8, dpi=canonical_dpi(profile))


def _figure_size(profile: str) -> tuple[float, float]:
    w, h = canonical_canvas_inches(profile)
    return w, h * 1.25


def _finalize_layout(fig: plt.Figure, profile: str = "web", hspace: float = 0.55) -> None:
    apply_review_layout(
        fig,
        mode=profile,
        left=0.08,
        right=0.97,
        bottom=0.12,
        top=0.88,
        wspace=0.48,
        hspace=hspace,
    )


def _shrink_axis_width(ax: plt.Axes, factor: float = 0.80) -> None:
    pos = ax.get_position()
    new_width = pos.width * factor
    ax.set_position([pos.x0, pos.y0, new_width, pos.height])


def _shift_axis_left(ax: plt.Axes, *, dx_px: float = 15.0) -> None:
    if dx_px <= 0:
        return
    fig = ax.figure
    if fig is None:
        return
    shift = dx_px / (fig.get_figwidth() * fig.dpi)
    pos = ax.get_position()
    ax.set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])


def _offset_axis(ax: plt.Axes, *, dx_px: float = 0.0, dy_px: float = 0.0) -> None:
    fig = ax.figure
    if fig is None:
        return
    shift_x = dx_px / (fig.get_figwidth() * fig.dpi)
    shift_y = dy_px / (fig.get_figheight() * fig.dpi)
    if shift_x == 0 and shift_y == 0:
        return
    pos = ax.get_position()
    ax.set_position([pos.x0 + shift_x, pos.y0 + shift_y, pos.width, pos.height])


def _shrink_waveguide_panel(
    ax: plt.Axes,
    *,
    width_factor: float = 1.0,
    height_factor: float = 1.0,
    y_shift: float = 0.0,
) -> None:
    pos = ax.get_position()
    new_width = pos.width * width_factor
    new_height = pos.height * height_factor
    new_y = pos.y0 + y_shift
    # Keep center alignment after vertical shrink to avoid clipping bottom labels.
    ax.set_position([pos.x0, new_y, new_width, new_height])


def _separate_horizontal_pair(
    left_ax: plt.Axes,
    right_ax: plt.Axes,
    *,
    gap_px: float = 10.0,
    width_factor: float = 0.90,
) -> None:
    """Shrink and separate two side-by-side panels to prevent overlap and edge crowding."""
    if left_ax.figure is None or right_ax.figure is None:
        return
    _shrink_axis_width(left_ax, factor=width_factor)
    _shrink_axis_width(right_ax, factor=width_factor)
    if gap_px <= 0:
        return
    shift = gap_px / (left_ax.figure.get_figwidth() * left_ax.figure.dpi)
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()
    left_ax.set_position([left_pos.x0 - shift / 2, left_pos.y0, left_pos.width, left_pos.height])
    right_ax.set_position(
        [right_pos.x0 + shift / 2, right_pos.y0, right_pos.width, right_pos.height]
    )


def _set_scenario_title(fig: plt.Figure, text: str, profile: str = "web") -> None:
    # Keep scenario banner consistently formatted and centrally aligned across profile.
    title_y = 0.972 if profile == "web" else 0.972
    set_tab_title(fig, text, mode=profile, y=title_y, fontsize=None)


def _note_axes(
    fig: plt.Figure, *, x: float = 0.08, y: float = 0.026, w: float = 0.84, h: float = 0.082
) -> plt.Axes:
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def _draw_summary_block(
    ax: plt.Axes,
    title: str,
    *,
    key_value_pairs: list[tuple[str, str]],
    note_lines: list[str],
    x_label: float = SUMMARY_LABEL_X,
    x_value: float = SUMMARY_VALUE_X,
    y_top: float = 0.98,
) -> None:
    y = y_top
    ax.text(
        x_label,
        y,
        title,
        transform=ax.transAxes,
        fontsize=SUMMARY_TITLE_FONT_SIZE,
        fontweight="bold",
        color=SERIES_PURPLE,
        va="top",
        ha="left",
    )
    y -= SUMMARY_LINE_GAP
    for label, value in key_value_pairs:
        ax.text(
            x_label,
            y,
            f"{label}:",
            transform=ax.transAxes,
            fontsize=SUMMARY_LABEL_FONT_SIZE,
            color=AXIS_COLOR,
            va="top",
            ha="left",
        )
        ax.text(
            x_value,
            y,
            value,
            transform=ax.transAxes,
            fontsize=SUMMARY_VALUE_FONT_SIZE,
            color=AXIS_COLOR,
            va="top",
            ha="left",
        )
        y -= SUMMARY_LINE_GAP
    y -= SUMMARY_LINE_GAP
    for line in note_lines:
        ax.text(
            x_label,
            y,
            line,
            transform=ax.transAxes,
            fontsize=SUMMARY_NOTE_FONT_SIZE,
            color=AXIS_COLOR,
            va="top",
            ha="left",
            linespacing=1.15,
        )
        y -= SUMMARY_NOTE_GAP


def _add_footnote(fig: plt.Figure, text: str, *, profile: str = "web") -> None:
    footnote_y = FOOTNOTE_Y_WEB if profile == "web" else FOOTNOTE_Y_PAPER
    rendered = textwrap.fill(text, width=250)
    fig.text(
        FOOTNOTE_X,
        footnote_y,
        rendered,
        transform=fig.transFigure,
        fontsize=FOOTNOTE_FONT_SIZE,
        color=AXIS_COLOR,
        va="top",
        ha="left",
        clip_on=False,
        wrap=True,
        linespacing=1.04,
    )


def _run_quantum(r: float, theta: float, eta_loss: float) -> tuple[np.ndarray, float, float, float]:
    res = run_single_mode(r, theta, eta_loss, X_VECTOR)
    intrinsic_sq_db = -10.0 * np.log10(np.exp(-2.0 * r)) if r > 0 else 0.0
    return res.W, res.var_x, res.var_p, intrinsic_sq_db


def _run_style_web_and_save(
    fig: plt.Figure,
    output_dir: pathlib.Path,
    name: str,
    profile: str,
    *,
    labels: Mapping[str, str],
    units: Mapping[str, str],
    notes: str,
    seed: int = 11,
    data_payload: Mapping[str, Any] | None = None,
) -> None:
    targets: list[pathlib.Path] = resolve_outputs(output_dir / name, profile)
    # preserve deterministic order and remove duplicates
    for target in list(dict.fromkeys(targets)):
        target_profile = target.parent.name
        dpi = canonical_dpi(target_profile)
        save_kwargs = {"dpi": dpi, "skip_tight_layout": True}
        if target_profile == "paper":
            save_ieee(fig, target.with_suffix(".pdf"), **save_kwargs)
            save_ieee(fig, target, dpi=dpi, skip_tight_layout=True)
        else:
            save_ieee(fig, target, dpi=dpi, skip_tight_layout=True)
        write_figure_meta(
            target,
            figure_id=target.stem,
            profile=target_profile,
            generator_script="scripts/generate_dashboard_gallery.py",
            generator_args=(f"--output-dir={output_dir}", f"--profile={target_profile}"),
            labels=labels,
            units=units,
            notes=notes,
            seed=seed,
            dpi=dpi,
            canvas_px=canonical_canvas_px(target_profile),
            data_payload=data_payload,
        )
    print(
        f"[OK] generated {len(list(dict.fromkeys(targets)))} files for {name}: "
        f"{[str(p) for p in dict.fromkeys(targets)]}"
    )


def _run_style_hardware(ax: plt.Axes, cfg: WaveguideConfig | None = None) -> None:
    if cfg is None:
        cfg = WaveguideConfig()
    n_eff, mode_area, ez_data, extent = run_hardware_simulation(cfg)
    ax.imshow(ez_data, extent=extent, cmap="RdBu", origin="lower", aspect="equal")
    ax.set_title("Waveguide mode profile", fontsize=AXIS_TITLE_FONT_SIZE)
    ax.set_xlabel("x (um)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("y (um)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.text(
        0.04,
        0.03,
        f"n_eff = {n_eff:.2f}\nArea = {mode_area:.2f} um^2",
        transform=ax.transAxes,
        fontsize=max(7, SUMMARY_LABEL_FONT_SIZE - 1),
        color=AXIS_COLOR,
        va="bottom",
        ha="left",
        clip_on=False,
    )
    style_axis(ax)


def _draw_calibration_curve(
    ax: plt.Axes,
    pump: float,
    *,
    title: str,
) -> tuple[float, float]:
    sq_powers = np.linspace(0.0, 500.0, 300)
    sq_db = -10.0 * np.log10(np.exp(-2.0 * calculate_squeezing(sq_powers)))
    r = calculate_squeezing(pump)
    observed = -10.0 * np.log10(np.exp(-2.0 * r)) if r > 0 else 0.0

    ax.plot(sq_powers, sq_db, color=SERIES_BLUE, lw=1.0)
    ax.axvline(pump, color=SERIES_ORANGE, ls="--", lw=0.85)
    ax.axhline(observed, color=SERIES_ORANGE, ls=":", lw=0.8, alpha=0.75)
    ax.scatter([pump], [observed], color=SERIES_ORANGE, s=34, zorder=5)
    style_axis(
        ax,
        title=title,
        xlabel="Pump power (mW)",
        ylabel="Intrinsic squeezing (dB)",
    )
    ax.set_xlim(0.0, 500.0)
    return observed, r


def _wigner_limit_for_r(r: float) -> float:
    # keep enough dynamic range for squeezed vs anti-squeezed components
    return float(np.clip(2.0 + 1.6 * max(0.0, r), 2.2, 4.0))


def _set_integer_axis_ticks(ax: plt.Axes) -> None:
    """Force axis ticks to integer values for easier review-readability."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _integer_limits_for_wigner(lim: float) -> np.ndarray:
    """Return symmetric integer ticks for Wigner panels."""
    if lim <= 0:
        lim = 1.0
    min_tick = int(np.floor(-lim))
    max_tick = int(np.ceil(lim))
    if min_tick > max_tick:
        min_tick, max_tick = -1, 1
    ticks = np.arange(min_tick, max_tick + 1, 1, dtype=float)
    if 0 not in ticks:
        ticks = np.array([0.0], dtype=float)
    return ticks


def _style_wigner_panel(
    ax: plt.Axes,
    frame: tuple[np.ndarray, float, float],
    title: str,
) -> None:
    image, r, intrinsic_sq_db = frame
    levels = np.linspace(-abs(image).max(), abs(image).max(), CONTOUR_LEVELS)
    cf = ax.contourf(
        X_VECTOR,
        X_VECTOR,
        image,
        levels=levels,
        cmap="RdBu_r",
    )
    colorbar = ax.figure.colorbar(cf, ax=ax, fraction=0.036, pad=0.05, shrink=0.88)
    colorbar.set_label("Wigner amplitude", fontsize=AXIS_LABEL_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=AXIS_LABEL_FONT_SIZE)
    colorbar.ax.yaxis.set_major_formatter(compact_axis_formatter())
    set_review_axis(
        ax,
        title=title,
        xlabel="x (SNU)",
        ylabel="p (SNU)",
        integer_ticks=True,
        xtick_format="%.2f",
    )
    ax.xaxis.set_major_formatter(compact_axis_formatter())
    ax.yaxis.set_major_formatter(compact_axis_formatter())
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    lim = _wigner_limit_for_r(r)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    try:
        ax.set_box_aspect(1.0)
    except AttributeError:
        # Matplotlib < 3.4 compatibility: rely on set_aspect with adjustable='box'
        pass
    ticks = _integer_limits_for_wigner(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def scenario_vacuum(profile: str, output_dir: pathlib.Path) -> None:
    _apply_style(profile)
    # Force 15x9 so layout maps 1:1 mathematically
    fig = plt.figure(figsize=(15.0, 9.0), dpi=100)

    try:
        from generate_prototypes import apply_prototype
    except ModuleNotFoundError:
        from scripts.generate_prototypes import apply_prototype

    W, var_x, var_p, _ = _run_quantum(0.0, 0.0, 1.0)
    pump = 0.0
    intrinsic_sq = 0.0

    title = f"Scenario 1: Vacuum baseline ({int(pump)} mW)"
    if profile == "paper":
        _set_scenario_title(fig, title, profile)
    footnote = (
        f"Scenario 1: Vacuum baseline ({int(pump)} mW) | Pump: {int(pump)} mW |\n"
        f"Intrinsic squeezing: {intrinsic_sq:.2f} dB | Var(x)=Var(p)=0.5 "
        f"(vac={var_x:.1f})\nVacuum reference scenario with pump power at 0 mW."
    )
    boxes = apply_prototype(
        fig,
        "dashboard_vacuum",
        profile,
        custom_title=title if profile == "web" else "",
        custom_footnote=footnote,
        hide_layout=(profile == "paper"),
    )

    ax_hw = fig.add_axes(boxes["Waveguide"])
    ax_cal = fig.add_axes(boxes["Calibration Curve"])
    ax_wig = fig.add_axes(boxes["Wigner"])
    ax_var = fig.add_axes(boxes["Photon Number"])

    _run_style_hardware(ax_hw, WaveguideConfig())
    _draw_calibration_curve(ax_cal, 0.0, title="Calibration curve")
    _style_wigner_panel(ax_wig, (W, 0.0, 0.0), "Wigner function: vacuum")

    metric_labels = ["Vx", "Vp", "SNU"]
    metric_positions = np.arange(len(metric_labels))
    bars = ax_var.bar(
        metric_positions,
        [var_x, var_p, 0.5],
        color=[SERIES_BLUE, SERIES_ORANGE, AXIS_COLOR],
        edgecolor=BG_COLOR,
        width=0.58,
    )
    for bar, value in zip(bars, [var_x, var_p, 0.5]):
        ax_var.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=SUMMARY_VALUE_FONT_SIZE,
            color=AXIS_COLOR,
            clip_on=False,
        )
    set_review_axis(
        ax_var,
        title="Quadrature variance",
        xlabel="Metric",
        ylabel="Variance (SNU)",
        integer_ticks=False,
        ystep=0.1,
        xticks=list(metric_positions),
    )
    ax_var.set_xticks(metric_positions)
    ax_var.set_xticklabels(metric_labels, rotation=0, ha="center")
    ax_var.set_ylim(0.0, 0.6)
    variance_ticks = np.arange(0.0, 0.61, 0.1)
    ax_var.set_yticks(variance_ticks)
    ax_var.set_yticklabels([f"{tick:.1f}" for tick in variance_ticks])
    _run_style_web_and_save(
        fig,
        output_dir,
        "dashboard_vacuum.png",
        profile,
        labels={"title": "Dashboard vacuum scenario", "xlabel": "x (SNU)", "ylabel": "p (SNU)"},
        units={
            "Var(x) [vac=0.5]": "Var(x) [vac=0.5]",
            "Var(p) [vac=0.5]": "Var(p) [vac=0.5]",
        },
        notes="Vacuum reference scenario with pump power at 0 mW.",
        data_payload={
            "var_x": np.array([var_x], dtype=float),
            "var_p": np.array([var_p], dtype=float),
            "loss_db": np.array([0.0], dtype=float),
            "intrinsic_sq_db_x": np.array([0.0], dtype=float),
            "observed_sq_db_x": np.array([0.0], dtype=float),
        },
    )
    plt.close(fig)


def scenario_calibration(profile: str, output_dir: pathlib.Path) -> None:
    _apply_style(profile)
    try:
        from generate_prototypes import apply_prototype
    except ModuleNotFoundError:
        from scripts.generate_prototypes import apply_prototype

    pump = 200.0
    r = calculate_squeezing(pump)
    intrinsic_sq = -10.0 * np.log10(np.exp(-2.0 * r))
    W, var_x, var_p, _ = _run_quantum(pump, 0.0, 1.0)
    max_n = 14

    # Force 15x9 so layout maps 1:1 mathematically
    fig = plt.figure(figsize=(15.0, 9.0), dpi=100)

    title = f"Scenario 2: Squeezed state ({pump:.0f} mW)"
    if profile == "paper":
        _set_scenario_title(fig, title, profile)
    footnote = (
        f"Scenario 2: Squeezed state ({int(pump)} mW) | Pump: {int(pump)} mW |\n"
        f"Intrinsic squeezing: {intrinsic_sq:.2f} dB | "
        f"Photon cutoff: n={max_n} | P(n) normalized.\nSqueezed-state calibration scenario using an intrinsic operating point and photon-number diagnostics."
    )
    boxes = apply_prototype(
        fig,
        "dashboard_calibration",
        profile,
        custom_title=title if profile == "web" else "",
        custom_footnote=footnote,
        hide_layout=(profile == "paper"),
    )

    ax_hw = fig.add_axes(boxes["Waveguide"])
    ax_cal = fig.add_axes(boxes["Calibration Curve"])
    ax_wig = fig.add_axes(boxes["Wigner"])
    ax_pn = fig.add_axes(boxes["Photon Number"])

    _run_style_hardware(ax_hw, WaveguideConfig())
    _draw_calibration_curve(ax_cal, pump, title="Calibration curve and operating point")
    _style_wigner_panel(
        ax_wig,
        (W, r, intrinsic_sq),
        f"Wigner function (r = {r:.3f})",
    )

    max_n = 14
    ns = np.arange(0, max_n + 1, 2)
    tanh_r = np.tanh(r)
    cosh_r = np.cosh(r)
    probs = []
    for n in ns:
        k = n // 2
        probs.append((factorial(n) / (factorial(k) ** 2 * 4**k)) * (tanh_r**n) / cosh_r)
    probs = np.array(probs, dtype=float)
    prob_sum = float(np.sum(probs))
    assert prob_sum > 0, "P(n) normalization sum is zero"
    probs = probs / prob_sum
    assert np.all(probs >= -1e-12), "negative probability detected in photon distribution"
    assert np.max(probs) <= 1.0 + 1e-6, "probabilities exceed 1"
    assert abs(np.sum(probs) - 1.0) <= 5e-2, (
        "P(n) should be normalized close to 1 in truncated basis"
    )

    colors = [SERIES_BLUE if n % 2 == 0 else AXIS_COLOR for n in ns]
    ax_pn.bar(ns, probs, color=colors, edgecolor=BG_COLOR, width=0.68)
    ax_pn.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_pn.set_xticks(ns)
    ax_pn.set_xlim(ns[0] - 0.6, ns[-1] + 0.6)
    ax_pn.set_ylim(0.0, 1.0)
    ax_pn.set_yticks(np.arange(0.0, 1.01, 0.1))
    set_review_axis(
        ax_pn,
        title="Photon number distribution P(n)",
        xlabel="Photon number n",
        ylabel="Probability",
        integer_ticks=True,
        ystep=0.1,
        xticks=[float(v) for v in ns],
        yticks=[float(v) for v in np.arange(0.0, 1.01, 0.1)],
    )
    ax_pn.grid(axis="y", alpha=0.25)

    ax_pn.grid(axis="y", alpha=0.25)
    _run_style_web_and_save(
        fig,
        output_dir,
        "dashboard_calibration.png",
        profile,
        labels={
            "title": "Dashboard squeezed-state calibration",
            "xlabel": "Pump power [mW]",
            "ylabel": "Squeezing X [dB]",
        },
        units={
            "Pump power [mW]": "Pump power [mW]",
            "Squeezing X [dB]": "Squeezing X [dB]",
            "Probability P(n) [-]": "Probability P(n) [-]",
        },
        notes="Squeezed-state calibration scenario using an intrinsic operating point and photon-number diagnostics.",
        data_payload={
            "pump_power_mw": np.array([pump], dtype=float),
            "var_x": np.array([var_x], dtype=float),
            "var_p": np.array([var_p], dtype=float),
            "p_n": probs,
            "p_n_sum": np.array([prob_sum], dtype=float),
            "intrinsic_sq_db_x": np.array([intrinsic_sq], dtype=float),
            "probability_sum": np.array([prob_sum], dtype=float),
        },
    )
    plt.close(fig)


def scenario_decoherence(profile: str, output_dir: pathlib.Path) -> None:
    _apply_style(profile)
    try:
        from generate_prototypes import apply_prototype
    except ModuleNotFoundError:
        from scripts.generate_prototypes import apply_prototype

    pump = 200.0
    r = calculate_squeezing(pump)
    loss_db_cm = 2.0
    length_mm = 5.0
    total_loss_db = loss_db_cm * (length_mm / 10.0)
    eta = float(db_to_eta(total_loss_db))

    W_pure, var_x_pure, var_p_pure, _ = _run_quantum(r, 0.0, 1.0)
    W_loss, var_x_loss, var_p_loss, _ = _run_quantum(r, 0.0, eta)

    intrinsic_sq_db = -10.0 * np.log10(np.exp(-2.0 * r))
    intrinsic_var_x = 0.5 * np.exp(-2.0 * r)
    obs_var_x = eta * intrinsic_var_x + (1.0 - eta) * 0.5
    obs_loss_db = -10.0 * np.log10(max(obs_var_x / 0.5, np.finfo(float).eps))
    assert obs_loss_db <= intrinsic_sq_db + 1e-9, (
        "Observed squeezing must not exceed intrinsic squeezing"
    )

    title = "Scenario 3: Decoherence + loss"
    footnote = (
        f"Scenario 3: Decoherence + loss | Pump power: {pump:.1f} mW | Squeezing parameter: r = {r:.4f} | Total transmissivity eta: {eta:.4f} |\n"
        f"Intrinsic (pre-loss): {intrinsic_sq_db:.2f} dB | Observed (post-loss): {obs_loss_db:.2f} dB | \n"
        "Loss moves squeezing toward vacuum limit (Var = 0.5). Intrinsic magnitude is always >= observed after loss."
    )

    # Force 15x9 so layout maps 1:1 mathematically
    fig = plt.figure(figsize=(15.0, 9.0), dpi=100)
    if profile == "paper":
        _set_scenario_title(fig, title, profile)

    boxes = apply_prototype(
        fig,
        "dashboard_decoherence",
        profile,
        custom_title=title if profile == "web" else "",
        custom_footnote=footnote,
        hide_layout=(profile == "paper"),
    )

    ax_w_pure = fig.add_axes(boxes["Wigner Pure"])
    ax_w_loss = fig.add_axes(boxes["Wigner Loss"])
    ax_var = fig.add_axes(boxes["Variance Sweep"])
    _style_wigner_panel(
        ax_w_pure,
        (W_pure, r, intrinsic_sq_db),
        "Wigner: loss = 0 dB",
    )
    _style_wigner_panel(
        ax_w_loss,
        (W_loss, r, intrinsic_sq_db),
        f"Wigner: loss = {total_loss_db:.1f} dB",
    )

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
        lw=1.25,
        label="Var(x) after fixed loss",
    )
    ax_var.axhline(0.5, color=AXIS_COLOR, ls="--", lw=0.85, label="Shot-noise limit")
    ax_var.axvline(pump, color=SERIES_ORANGE, ls="--", lw=0.85)
    ax_var.scatter([pump], [obs_var_x], color=SERIES_TEAL, s=35)
    style_axis(
        ax_var,
        title="Squeezing after fixed loss vs pump power",
        xlabel="Pump power P (mW)",
        ylabel="Var(x) (SNU; vacuum=0.5)",
    )
    ax_var.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_var.set_xlim(0.0, 500.0)
    ax_var.set_xticks(np.linspace(0.0, 500.0, 6))
    ax_var.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_var.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_var.set_yticks(np.arange(0.10, 0.53, 0.1))
    ax_var.set_ylim(0.10, 0.52)
    ax_var.legend(loc="lower right")
    ax_var.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_var.tick_params(axis="y", which="major", length=5, width=1.0)
    ax_var.tick_params(axis="y", which="minor", length=2.5, width=0.7)
    ax_var.grid(axis="y", which="major", alpha=0.30)
    ax_var.grid(axis="y", which="minor", alpha=0.12, linestyle=":")

    _run_style_web_and_save(
        fig,
        output_dir,
        "dashboard_decoherence.png",
        profile,
        labels={
            "title": "Dashboard decoherence scenario",
            "xlabel": "Pump power [mW]",
            "ylabel": "Squeezing X [dB]",
        },
        units={
            "Pump power [mW]": "Pump power [mW]",
            "Squeezing X [dB]": "Squeezing X [dB]",
            "Loss [dB]": "Loss [dB]",
        },
        notes="Decoherence scenario compares intrinsic and observed squeezing under fixed loss.",
        data_payload={
            "pump_power_mw": np.array([pump], dtype=float),
            "loss_db": np.array([total_loss_db], dtype=float),
            "intrinsic_sq_db_x": np.array([intrinsic_sq_db], dtype=float),
            "observed_sq_db_x": np.array([obs_loss_db], dtype=float),
            "eta": np.array([eta], dtype=float),
        },
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ASSETS_DIR,
        help="Directory where output PNGs are written.",
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

    print("Generating dashboard scenario gallery ...\n")
    for profile in profiles:
        scenario_vacuum(profile, output_dir)
        scenario_calibration(profile, output_dir)
        scenario_decoherence(profile, output_dir)
    print(f"\nDone. Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
