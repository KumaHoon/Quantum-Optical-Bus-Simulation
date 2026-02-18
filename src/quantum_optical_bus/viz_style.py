"""Compatibility wrapper over IEEE plotting helpers."""

from __future__ import annotations

import pathlib
import sys

from matplotlib.figure import Figure

_ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from scripts.viz_style_ieee import (  # type: ignore
    AXIS_COLOR,
    BG_COLOR,
    BASE_FONT_SIZE,
    FIGURE_WIDTH_2COL_IN,
    IEEE_STYLE_CYCLE,
    PANEL_COLOR,
    SERIES_BLUE,
    SERIES_GRAY,
    SERIES_ORANGE,
    SERIES_PURPLE,
    SERIES_TEAL,
    SMALL_FONT_SIZE,
    SUPTITLE_FONT_SIZE,
    TICK_FONT_SIZE,
    TEXT_COLOR,
    TITLE_FONT_SIZE,
    apply_ieee_axes,
    apply_ieee_style,
    apply_layout,
    ieee_figsize,
    save_ieee,
    style_axis,
)

__all__ = [
    "AXIS_COLOR",
    "BG_COLOR",
    "BASE_FONT_SIZE",
    "Figure",
    "FIGURE_WIDTH_2COL_IN",
    "IEEE_STYLE_CYCLE",
    "PANEL_COLOR",
    "SERIES_BLUE",
    "SERIES_GRAY",
    "SERIES_ORANGE",
    "SERIES_PURPLE",
    "SERIES_TEAL",
    "TICK_FONT_SIZE",
    "TITLE_FONT_SIZE",
    "SUPTITLE_FONT_SIZE",
    "TEXT_COLOR",
    "SMALL_FONT_SIZE",
    "apply_ieee_axes",
    "apply_ieee_style",
    "apply_layout",
    "ieee_figsize",
    "save_ieee",
    "style_axis",
]
