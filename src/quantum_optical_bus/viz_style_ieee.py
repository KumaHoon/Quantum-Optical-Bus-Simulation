"""Shared IEEE-like Matplotlib styling helpers.

This module centralizes figure styling so scripts and app code stay visually
consistent. It intentionally focuses on a conservative style profile that uses a
color/marker/linestyle cycle friendly to color-blind viewers and avoids
red-green-dependent semantics.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib import cycler, rcParams
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
GRID_COLOR = "#30363d"
AXIS_COLOR = "#8b949e"
TEXT_COLOR = "#c9d1d9"

SERIES_BLUE = "#1f77b4"
SERIES_ORANGE = "#ff7f0e"
SERIES_TEAL = "#17becf"
SERIES_PURPLE = "#9467bd"
SERIES_BROWN = "#8c564b"
SERIES_GRAY = "#7f7f7f"

BASE_FONT_SIZE = 10
TICK_FONT_SIZE = 9
TITLE_FONT_SIZE = 12
SUPTITLE_FONT_SIZE = 12
SMALL_FONT_SIZE = 8

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
    dpi: int = 300,
) -> None:
    """Apply repository-wide IEEE-like Matplotlib defaults."""

    if title_font_size > 12:
        title_font_size = 12
    if suptitle_font_size > 12:
        suptitle_font_size = 12

    rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": PANEL_COLOR,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlesize": title_font_size,
            "axes.titleweight": "bold",
            "axes.titlepad": 6,
            "axes.labelsize": base_font_size,
            "axes.labelpad": 4,
            "text.color": TEXT_COLOR,
            "font.size": base_font_size,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
                "sans-serif",
            ],
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.fontsize": SMALL_FONT_SIZE,
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.handlelength": 1.5,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.grid": True,
        }
    )

    style_cycle = get_ieee_style_cycle()
    rcParams["axes.prop_cycle"] = cycler(
        color=[c for c, _marker, _linestyle in style_cycle],
        marker=[_marker for _c, _marker, _linestyle in style_cycle],
        linestyle=[_linestyle for _c, _marker, _linestyle in style_cycle],
    )


def get_ieee_style_cycle() -> list[tuple[str, str, str]]:
    """Return a safe color/marker/linestyle cycle definition."""

    return list(IEEE_STYLE_CYCLE)


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
) -> None:
    """Backward-compatible name for applying the standard rc parameters."""

    set_ieee_rcparams(
        base_font_size=base_font_size,
        tick_font_size=tick_font_size,
        dpi=dpi,
    )


def apply_ieee_axes(ax: plt.Axes, xlabel: str, ylabel: str, *, title: str | None = None) -> None:
    """Style axis labels and ticks in the shared IEEE-like style."""

    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    if title:
        ax.set_title(title)

    ax.set_facecolor(PANEL_COLOR)
    ax.grid(True, alpha=0.32)

    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)

    ax.tick_params(axis="both", colors=AXIS_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def style_axis(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Compatibility wrapper for older axis formatting call sites."""

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_facecolor(PANEL_COLOR)
    ax.grid(axis="both", alpha=0.28)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.tick_params(axis="both", colors=AXIS_COLOR)


def save_ieee(fig: Figure, path: str | Path, *, dpi: int = 300) -> None:
    """Save with clipping-safe defaults."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.set_constrained_layout(False)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.98))
    fig.savefig(out, dpi=dpi, pad_inches=0.02)


def apply_layout(fig: Figure, *, use_constrained: bool = True) -> None:
    """Compatibility helper used by existing generators."""

    if use_constrained:
        fig.set_constrained_layout(False)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))


__all__ = [
    "AXIS_COLOR",
    "BG_COLOR",
    "BASE_FONT_SIZE",
    "FIGURE_WIDTH_2COL_IN",
    "GRID_COLOR",
    "IEEE_STYLE_CYCLE",
    "PANEL_COLOR",
    "SERIES_BLUE",
    "SERIES_BROWN",
    "SERIES_ORANGE",
    "SERIES_PURPLE",
    "SERIES_TEAL",
    "SERIES_GRAY",
    "SMALL_FONT_SIZE",
    "SUPTITLE_FONT_SIZE",
    "TICK_FONT_SIZE",
    "TEXT_COLOR",
    "TITLE_FONT_SIZE",
    "get_ieee_style_cycle",
    "set_ieee_rcparams",
    "apply_ieee_style",
    "ieee_figsize",
    "apply_ieee_axes",
    "apply_layout",
    "save_ieee",
    "style_axis",
]
