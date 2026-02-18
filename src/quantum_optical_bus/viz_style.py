"""Shared matplotlib styling helpers for IEEE-like dashboard figures."""

from __future__ import annotations

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Dark dashboard palette
BG_COLOR = "#0d1117"
PANEL_COLOR = "#161b22"
GRID_COLOR = "#30363d"
AXIS_COLOR = "#8b949e"
TEXT_COLOR = "#c9d1d9"

# Series palette avoids red/green pairing.
SERIES_BLUE = "#58a6ff"
SERIES_ORANGE = "#f0ad00"
SERIES_PURPLE = "#d2a8ff"
SERIES_TEAL = "#56d4dd"
SERIES_GRAY = "#8b949e"

BASE_FONT_SIZE = 10
TICK_FONT_SIZE = 9
TITLE_FONT_SIZE = 11
SUPTITLE_FONT_SIZE = 13

# IEEE-like target width for 2-column figures (inches).
FIGURE_WIDTH_2COL_IN = 7.16


def apply_ieee_style(
    *,
    base_font_size: int = BASE_FONT_SIZE,
    tick_font_size: int = TICK_FONT_SIZE,
    dpi: int = 300,
) -> None:
    """Apply a shared, readable dark style with conservative typography."""
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
            "text.color": TEXT_COLOR,
            "font.size": base_font_size,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
            "xtick.labelsize": tick_font_size,
            "ytick.labelsize": tick_font_size,
            "legend.fontsize": 8,
            "legend.edgecolor": AXIS_COLOR,
            "legend.fancybox": False,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.35,
        }
    )


def style_axis(
    ax,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Apply consistent axis styling (ticks, grid, labels)."""
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


def apply_layout(fig: Figure, *, use_constrained: bool = True) -> None:
    """Apply a stable layout pass with reserved top margin for suptitle."""
    if use_constrained:
        try:
            fig.set_constrained_layout(False)
            fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
            return
        except Exception:
            pass
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
