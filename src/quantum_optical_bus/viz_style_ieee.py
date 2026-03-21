"""Shared IEEE-like Matplotlib styling helpers.

This module centralizes figure styling so scripts and app code stay visually
consistent.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from matplotlib import cycler, rcParams
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, FormatStrFormatter, ScalarFormatter


BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
GRID_COLOR = "#30363d"
AXIS_COLOR = "#8b949e"
TEXT_COLOR = "#c9d1d9"

PAPER_BG_COLOR = "#ffffff"
PAPER_PANEL_COLOR = "#ffffff"
PAPER_GRID_COLOR = "#d2dce6"
PAPER_AXIS_COLOR = "#374151"
PAPER_TEXT_COLOR = "#0f172a"
PAPER_LEGEND_FONT_SIZE = 9

SERIES_BLUE = "#1f77b4"
SERIES_ORANGE = "#ff7f0e"
SERIES_TEAL = "#17becf"
SERIES_PURPLE = "#9467bd"
SERIES_BROWN = "#8c564b"
SERIES_GRAY = "#7f7f7f"

BASE_FONT_SIZE = 12
TICK_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
SUPTITLE_FONT_SIZE = 12
PAPER_TITLE_FONT_SIZE = 11
PAPER_SUPTITLE_FONT_SIZE = 11
SMALL_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 12
AXIS_TITLE_FONT_SIZE = 12
TAB_TITLE_SIZE_WEB = 12
TAB_TITLE_SIZE_PAPER = 11
TAB_TITLE_Y = 0.985

PAPER_TEXT_FALLBACK = 10
WEB_LINE_WIDTH = 2.0
WEB_MARKER_SIZE = 6.0
PAPER_LINE_WIDTH = 1.5
PAPER_MARKER_SIZE = 4.0

FIGURE_WIDTH_2COL_IN = 7.16


IEEE_STYLE_CYCLE = [
    (SERIES_BLUE, "o", "-"),
    (SERIES_ORANGE, "s", "--"),
    (SERIES_TEAL, "^", ":"),
    (SERIES_PURPLE, "D", "-."),
    (SERIES_BROWN, "v", "-"),
    (SERIES_GRAY, "x", "--"),
]


def set_ieee_rcparams(
    *,
    base_font_size: int = BASE_FONT_SIZE,
    tick_font_size: int = TICK_FONT_SIZE,
    title_font_size: int = TITLE_FONT_SIZE,
    suptitle_font_size: int = SUPTITLE_FONT_SIZE,
    legend_font_size: int = SMALL_FONT_SIZE,
    dpi: int = 300,
    profile: str = "web",
) -> None:
    """Apply repository-wide IEEE-like Matplotlib defaults."""

    if profile not in {"web", "paper"}:
        raise ValueError("profile must be 'web' or 'paper'")

    is_web = profile == "web"
    fig_bg = BG_COLOR if is_web else PAPER_BG_COLOR
    panel_bg = PANEL_COLOR if is_web else PAPER_PANEL_COLOR
    axis_color = AXIS_COLOR if is_web else PAPER_AXIS_COLOR
    text_color = TEXT_COLOR if is_web else PAPER_TEXT_COLOR
    grid_color = GRID_COLOR if is_web else PAPER_GRID_COLOR

    # Contract-aware minimums/defaults:
    if is_web:
        base_font_size = max(base_font_size, 10)
        tick_font_size = max(tick_font_size, 8)
        title_font_size = max(title_font_size, 10)
        suptitle_font_size = max(suptitle_font_size, 10)
        suptitle_font_size = min(suptitle_font_size, 20)
        title_font_size = min(title_font_size, 20)
        legend_font_size = max(legend_font_size, 8)
        line_width = WEB_LINE_WIDTH
        marker_size = WEB_MARKER_SIZE
    else:
        base_font_size = min(max(base_font_size, 9), 10)
        tick_font_size = min(max(tick_font_size, 8), 9)
        title_font_size = min(max(title_font_size, 10), 12)
        suptitle_font_size = min(max(suptitle_font_size, 10), 12)
        legend_font_size = min(max(legend_font_size, 8), 9)
        line_width = PAPER_LINE_WIDTH
        marker_size = PAPER_MARKER_SIZE

    rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": fig_bg,
            "savefig.facecolor": fig_bg,
            "axes.facecolor": panel_bg,
            "axes.edgecolor": axis_color,
            "axes.labelcolor": text_color,
            "axes.titlesize": title_font_size,
            "axes.titleweight": "bold",
            "axes.titlepad": 6,
            "axes.labelsize": base_font_size,
            "axes.labelpad": 4,
            "text.color": text_color,
            "font.size": base_font_size,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
                "sans-serif",
            ],
            "xtick.color": axis_color,
            "ytick.color": axis_color,
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.fontsize": legend_font_size,
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.handlelength": 1.5,
            "grid.color": grid_color,
            "grid.alpha": 0.35,
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "axes.grid": True,
        }
    )

    rcParams["axes.prop_cycle"] = cycler(
        color=[c for c, _marker, _linestyle in IEEE_STYLE_CYCLE],
        marker=[_marker for _c, _marker, _linestyle in IEEE_STYLE_CYCLE],
        linestyle=[_linestyle for _c, _marker, _linestyle in IEEE_STYLE_CYCLE],
    )


def get_ieee_style_cycle() -> list[tuple[str, str, str]]:
    """Return a safe color/marker/linestyle cycle definition."""

    return list(IEEE_STYLE_CYCLE)


def apply_figstyle(
    *,
    profile: str = "web",
    base_font_size: int = BASE_FONT_SIZE,
    tick_font_size: int = TICK_FONT_SIZE,
    dpi: int = 300,
    title_font_size: int | None = None,
    legend_font_size: int | None = None,
    annotation_font_size: int | None = None,
) -> None:
    """Apply IEEE-like style presets for a rendering profile."""

    if profile == "paper":
        rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [
                    "Times New Roman",
                    "Times",
                    "STIXGeneral",
                    "serif",
                ],
            }
        )
        set_ieee_rcparams(
            base_font_size=max(base_font_size, 9),
            tick_font_size=max(tick_font_size, 8),
            title_font_size=title_font_size or PAPER_TITLE_FONT_SIZE,
            suptitle_font_size=PAPER_SUPTITLE_FONT_SIZE,
            legend_font_size=legend_font_size or PAPER_LEGEND_FONT_SIZE,
            dpi=dpi,
            profile=profile,
        )
        return

    set_ieee_rcparams(
        base_font_size=title_font_size or base_font_size,
        tick_font_size=tick_font_size,
        title_font_size=title_font_size or TITLE_FONT_SIZE,
        suptitle_font_size=SUPTITLE_FONT_SIZE,
        legend_font_size=legend_font_size or SMALL_FONT_SIZE,
        dpi=dpi,
        profile="web",
    )


def apply_web_style(
    *,
    base_font_size: int = BASE_FONT_SIZE,
    tick_font_size: int = TICK_FONT_SIZE,
    dpi: int = 300,
) -> None:
    """Apply the dark web profile."""

    apply_figstyle(
        profile="web", base_font_size=base_font_size, tick_font_size=tick_font_size, dpi=dpi
    )


def apply_paper_style(
    *,
    base_font_size: int = BASE_FONT_SIZE,
    tick_font_size: int = TICK_FONT_SIZE,
    dpi: int = 300,
) -> None:
    """Apply the paper/print profile."""

    apply_figstyle(
        profile="paper", base_font_size=base_font_size, tick_font_size=tick_font_size, dpi=dpi
    )


def ieee_figsize(
    *, width_in: float = FIGURE_WIDTH_2COL_IN, aspect: float = 0.62
) -> tuple[float, float]:
    """Return a standard IEEE-leaning figure size in inches."""

    return width_in, width_in * aspect


def apply_ieee_style(
    *,
    base_font_size: int = BASE_FONT_SIZE,
    tick_font_size: int = TICK_FONT_SIZE,
    dpi: int = 300,
    profile: str = "web",
) -> None:
    """Backward-compatible name for applying the standard rc parameters."""

    set_ieee_rcparams(
        base_font_size=base_font_size,
        tick_font_size=tick_font_size,
        dpi=dpi,
        profile=profile,
    )


def apply_ieee_axes(ax: plt.Axes, xlabel: str, ylabel: str, *, title: str | None = None) -> None:
    """Style axis labels and ticks in the shared IEEE-like style."""

    face_color = rcParams.get("axes.facecolor", PANEL_COLOR)
    axis_color = rcParams.get("axes.edgecolor", AXIS_COLOR)
    text_color = rcParams.get("axes.labelcolor", TEXT_COLOR)
    grid_color = rcParams.get("grid.color", GRID_COLOR)
    tick_color = rcParams.get("xtick.color", AXIS_COLOR)

    label_size = float(rcParams.get("axes.labelsize", AXIS_LABEL_FONT_SIZE))
    tick_size = float(rcParams.get("xtick.labelsize", TICK_FONT_SIZE))
    title_size = float(rcParams.get("axes.titlesize", AXIS_TITLE_FONT_SIZE))
    title_color = rcParams.get("text.color", text_color)

    ax.set_xlabel(xlabel, labelpad=5, fontsize=label_size)
    ax.set_ylabel(ylabel, labelpad=5, fontsize=label_size)
    if title:
        ax.set_title(title, color=title_color, fontsize=title_size)

    ax.set_facecolor(face_color)
    ax.grid(True, alpha=0.32, color=grid_color)

    for spine in ax.spines.values():
        spine.set_color(axis_color)

    ax.tick_params(axis="both", colors=tick_color, labelsize=tick_size, pad=4)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)


def apply_review_layout(
    fig: Figure,
    *,
    mode: str = "web",
    left: float = 0.07,
    right: float = 0.98,
    bottom: float = 0.10,
    top: float = 0.94,
    wspace: float = 0.24,
    hspace: float = 0.18,
) -> None:
    """Apply a robust layout for quick review inspection."""

    if mode not in {"web", "paper"}:
        raise ValueError("mode must be 'web' or 'paper'")

    fig.set_constrained_layout(False)

    if mode == "web":
        left = max(left, 0.08)
        right = min(right, 0.98)
        bottom = max(bottom, 0.08)
        top = min(top, 0.95)
    else:
        left = max(left, 0.09)
        right = min(right, 0.97)
        bottom = max(bottom, 0.09)
        top = min(top, 0.93)

    if left >= right - 0.25:
        # Prevent inverted/squeezed layouts and keep a readable plot envelope.
        left = 0.08
        right = 0.96
    if bottom >= top - 0.20:
        bottom = 0.10
        top = 0.90

    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )


def set_tab_title(
    fig: Figure,
    text: str,
    *,
    mode: str = "web",
    fontsize: int | None = None,
    y: float | None = None,
) -> None:
    """Apply a unified figure title format for all top-level tabs/figures."""
    if mode not in {"web", "paper"}:
        raise ValueError("mode must be 'web' or 'paper'")
    if y is None:
        y = TAB_TITLE_Y
    if fontsize is None:
        fontsize = TAB_TITLE_SIZE_WEB
    fig.suptitle(
        text,
        x=0.5,
        y=y,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=plt.rcParams.get("text.color", TEXT_COLOR),
        transform=fig.transFigure,
        clip_on=False,
        wrap=True,
    )


def set_review_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    integer_ticks: bool = False,
    ystep: float | None = None,
    xstep: float | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_format: str | None = None,
    ytick_format: str | None = None,
) -> None:
    """Apply readable axis labels/ticks for review-oriented figures."""

    if integer_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if ystep is not None:
            ax.yaxis.set_major_locator(MaxNLocator(integer=False))
            y_min, y_max = ax.get_ylim()
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                ticks = np.arange(
                    np.floor(y_min / ystep) * ystep,
                    np.ceil(y_max / ystep) * ystep + 1e-9,
                    ystep,
                )
                ax.set_yticks(ticks)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        if xstep is not None:
            x_min, x_max = ax.get_xlim()
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                start = np.floor(x_min / xstep) * xstep
                stop = np.ceil(x_max / xstep) * xstep + 1e-9
                ax.set_xticks(np.arange(start, stop, xstep))
        elif xticks is not None:
            ax.set_xticks(xticks)

    if ystep is not None and not integer_ticks:
        y_min, y_max = ax.get_ylim()
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            ticks = np.arange(
                np.floor(y_min / ystep) * ystep,
                np.ceil(y_max / ystep) * ystep + 1e-9,
                ystep,
            )
            ax.set_yticks(ticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if xticks is not None:
        ax.set_xticks(xticks)

    apply_ieee_axes(ax, xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel())
    if title is not None:
        ax.set_title(title, fontsize=float(rcParams.get("axes.titlesize", AXIS_TITLE_FONT_SIZE)))
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=float(rcParams.get("axes.labelsize", AXIS_LABEL_FONT_SIZE)))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=float(rcParams.get("axes.labelsize", AXIS_LABEL_FONT_SIZE)))
    ax.tick_params(
        axis="both", pad=4, labelsize=float(rcParams.get("xtick.labelsize", TICK_FONT_SIZE))
    )
    if xtick_format is not None:
        ax.xaxis.set_major_formatter(FormatStrFormatter(xtick_format))
    if ytick_format is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(ytick_format))
    if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter) and isinstance(
        ax.yaxis.get_major_formatter(), ScalarFormatter
    ):
        ax.ticklabel_format(style="plain")


def compact_axis_tick_formatter(
    value: float,
    pos: float | None = None,
    decimals: int = 2,
    epsilon: float = 1e-9,
) -> str:
    """Return integer-like ticks as integers, keep decimals otherwise."""

    if np.isfinite(value) and abs(value - np.round(value)) <= epsilon:
        return f"{int(round(value))}"
    return f"{value:.{decimals}f}"


def compact_axis_formatter(*, decimals: int = 2, epsilon: float = 1e-9) -> FuncFormatter:
    """Formatter used for axes where 0.00 should be shown as 0."""

    return FuncFormatter(
        lambda value, pos: compact_axis_tick_formatter(
            value,
            pos=pos,
            decimals=decimals,
            epsilon=epsilon,
        )
    )


def style_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Compatibility wrapper for older axis formatting call sites."""

    axis_label_size = float(rcParams.get("axes.labelsize", AXIS_LABEL_FONT_SIZE))
    if title is not None:
        ax.set_title(title, fontsize=float(rcParams.get("axes.titlesize", AXIS_TITLE_FONT_SIZE)))
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=8, fontsize=axis_label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=8, fontsize=axis_label_size)
    apply_ieee_axes(ax, xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), title=None)


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    profile: str | None = None,
    skip_tight_layout: bool = False,
) -> None:
    """Persist a figure using IEEE-style defaults.

    ``profile`` is optional and allows inline style clarification at call sites.
    """

    if profile is not None:
        apply_figstyle(profile=profile)
    save_ieee(fig, path, dpi=dpi, skip_tight_layout=skip_tight_layout)


def save_ieee(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    skip_tight_layout: bool = False,
) -> None:
    """Save with clipping-safe defaults."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.set_constrained_layout(False)
    if not skip_tight_layout:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
            )
            fig.tight_layout(rect=(0.03, 0.08, 0.97, 0.94), pad=1.15)
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
        return

    # PNG/PDF preview mode: keep figures on the canvas and detached from borders.
    fig.savefig(out, dpi=dpi, bbox_inches=None, pad_inches=0.08)


def apply_layout(fig: Figure, *, use_constrained: bool = True) -> None:
    """Compatibility helper used by existing generators."""

    if use_constrained:
        fig.set_constrained_layout(False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
        )
        fig.tight_layout(rect=(0.03, 0.06, 0.97, 0.94), pad=1.0)


__all__ = [
    "AXIS_COLOR",
    "BG_COLOR",
    "BASE_FONT_SIZE",
    "FIGURE_WIDTH_2COL_IN",
    "GRID_COLOR",
    "IEEE_STYLE_CYCLE",
    "PANEL_COLOR",
    "PAPER_BG_COLOR",
    "PAPER_PANEL_COLOR",
    "PAPER_AXIS_COLOR",
    "PAPER_GRID_COLOR",
    "PAPER_TEXT_COLOR",
    "SERIES_BLUE",
    "SERIES_BROWN",
    "SERIES_ORANGE",
    "SERIES_PURPLE",
    "SERIES_TEAL",
    "SERIES_GRAY",
    "SMALL_FONT_SIZE",
    "SUPTITLE_FONT_SIZE",
    "TEXT_COLOR",
    "TITLE_FONT_SIZE",
    "TICK_FONT_SIZE",
    "compact_axis_formatter",
    "compact_axis_tick_formatter",
    "apply_review_layout",
    "apply_ieee_axes",
    "apply_ieee_style",
    "apply_figstyle",
    "apply_layout",
    "set_review_axis",
    "set_tab_title",
    "apply_paper_style",
    "apply_web_style",
    "get_ieee_style_cycle",
    "ieee_figsize",
    "save_ieee",
    "save_figure",
    "set_ieee_rcparams",
    "style_axis",
]
