"""IEEE-style Matplotlib helpers for dashboard and publication figures."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rcParams


BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
GRID_COLOR = "#30363d"
AXIS_COLOR = "#8b949e"
TEXT_COLOR = "#c9d1d9"

SERIES_BLUE = "#58a6ff"
SERIES_ORANGE = "#f0ad00"
SERIES_PURPLE = "#d2a8ff"
SERIES_TEAL = "#56d4dd"
SERIES_GRAY = "#8b949e"

BASE_FONT_SIZE = 10
TICK_FONT_SIZE = 9
TITLE_FONT_SIZE = 11
SUPTITLE_FONT_SIZE = 12
SMALL_FONT_SIZE = 8

FIGURE_WIDTH_2COL_IN = 7.16

IEEE_STYLE_CYCLE = [
    (SERIES_BLUE, "o", "-"),
    (SERIES_ORANGE, "s", "--"),
    (SERIES_TEAL, "^", ":"),
    (SERIES_PURPLE, "D", "-"),
    (SERIES_GRAY, "x", "-"),
]


def ieee_figsize(*, width_in: float = FIGURE_WIDTH_2COL_IN, aspect: float = 0.62) -> tuple[float, float]:
    """Return a 2-column IEEE-friendly figure size."""

    return (width_in, width_in * aspect)


def apply_ieee_style(
    *,
    base_font_size: int = BASE_FONT_SIZE,
    tick_font_size: int = TICK_FONT_SIZE,
    dpi: int = 300,
) -> None:
    """Apply a consistent IEEE-style dark plotting style."""

    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": PANEL_COLOR,
            "axes.edgecolor": AXIS_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.titleweight": "bold",
            "axes.titlepad": 6,
            "axes.labelsize": base_font_size,
            "text.color": TEXT_COLOR,
            "font.size": base_font_size,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.handlelength": 1.6,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.grid": True,
            "axes.labelpad": 4,
        }
    )
    rcParams["axes.prop_cycle"] = plt.cycler(color=[c for c, *_ in IEEE_STYLE_CYCLE])


def apply_ieee_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    *,
    title: str | None = None,
    unit_hint: str | None = None,
) -> None:
    """Apply standard axis labels/titles for consistent figure language."""

    ax.set_xlabel(_format_axis_text(xlabel, unit_hint), labelpad=5)
    ax.set_ylabel(_format_axis_text(ylabel), labelpad=5)
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
    """Backward-compatible wrapper for legacy axis styling."""

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
    """Save with conservative clipping-safe defaults."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.98))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")


def apply_layout(fig: Figure, *, use_constrained: bool = True) -> None:
    """Apply a reliable layout pass."""

    if use_constrained:
        fig.set_constrained_layout(False)
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    else:
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))


def _format_axis_text(label: str, unit_hint: str | None = None) -> str:
    if unit_hint and "(" not in label:
        return f"{label} ({unit_hint})"
    return label
