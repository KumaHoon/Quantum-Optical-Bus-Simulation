"""Repository-level figure styling + artifact metadata helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from figure_style_contract import (
        canonical_canvas_inches,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_artifacts,
    )
except ModuleNotFoundError:
    from scripts.figure_style_contract import (
        canonical_canvas_inches,
        canonical_canvas_px,
        canonical_dpi,
        write_figure_artifacts,
    )
from quantum_optical_bus.viz_style_ieee import (
    AXIS_LABEL_FONT_SIZE,
    AXIS_TITLE_FONT_SIZE,
    apply_paper_style,
    apply_web_style,
)
import matplotlib.pyplot as plt


def apply_style(
    profile: str,
    *,
    base_font_size: int | None = None,
    tick_font_size: int | None = None,
    dpi: int | None = None,
) -> int:
    """Apply profile-specific style and return dpi used.

    Parameters
    ----------
    profile:
        Rendering profile. Must be ``web`` or ``paper``.
    base_font_size:
        Optional override for axis/title base font size.
    tick_font_size:
        Optional override for tick font size.
    dpi:
        Optional override for output DPI.
    """

    profile_dpi = canonical_dpi(profile) if dpi is None else dpi
    if profile == "paper":
        apply_paper_style(
            base_font_size=11 if base_font_size is None else base_font_size,
            tick_font_size=10 if tick_font_size is None else tick_font_size,
            dpi=profile_dpi,
        )
        # Match web typography scale for figure titles and axis labels in paper mode.
        plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONT_SIZE
        plt.rcParams["axes.titlesize"] = AXIS_TITLE_FONT_SIZE
        return profile_dpi
    if profile == "web":
        apply_web_style(
            base_font_size=10 if base_font_size is None else base_font_size,
            tick_font_size=9 if tick_font_size is None else tick_font_size,
            dpi=profile_dpi,
        )
        return profile_dpi
    raise ValueError(f"Unsupported profile '{profile}', expected 'web' or 'paper'.")


def write_figure_meta(
    path: Path,
    *,
    profile: str,
    generator_script: str,
    generator_args: Sequence[str] = (),
    labels: Mapping[str, str],
    units: Mapping[str, str],
    notes: str | Sequence[str],
    figure_id: str | None = None,
    seed: int | None = None,
    canvas_px: tuple[int, int] | None = None,
    dpi: int | None = None,
    data_payload: Mapping[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    """Emit metadata via the shared contract helper.

    ``args`` and ``generator_args`` are both recorded for compatibility with
    existing tooling and the figure-contract terminology.
    """

    return write_figure_artifacts(
        path,
        figure_id=(path.stem if figure_id is None else figure_id),
        profile=profile,
        generator_script=generator_script,
        generator_args=generator_args,
        labels=dict(labels),
        units=dict(units),
        notes=notes,
        seed=seed,
        canvas_px=canonical_canvas_px(profile) if canvas_px is None else canvas_px,
        dpi=canonical_dpi(profile) if dpi is None else dpi,
        payload_extra={"args": tuple(generator_args)},
        data_payload=data_payload,
    )


__all__ = [
    "apply_style",
    "write_figure_meta",
    "canonical_canvas_inches",
    "canonical_canvas_px",
    "canonical_dpi",
]
