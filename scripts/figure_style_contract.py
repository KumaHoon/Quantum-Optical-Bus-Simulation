"""Figure style contract helpers for generated artifacts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

WEB_CANVAS_PX: tuple[int, int] = (1600, 1000)
WEB_DPI: int = 150
PAPER_CANVAS_PX: tuple[int, int] = (2100, 1350)
PAPER_DPI: int = 300

WEB_LABEL = "web"
PAPER_LABEL = "paper"

REQUIRED_META_KEYS = {
    "figure_id",
    "profile",
    "created_at_utc",
    "generator_script",
    "args",
    "generator_args",
    "labels",
    "units",
    "canvas_px",
    "dpi",
    "notes",
    "style",
    "git_commit",
    "python_version",
}


FONT_STYLE_MIN_MAX = {
    WEB_LABEL: {
        "title_font_size": (16, 20),
        "axis_label_font_size": (14, 16),
        "tick_font_size": (12, 14),
        "legend_font_size": (12, 14),
        "annotation_font_size": (12, 99),
        "line_width": (2.0, 2.0),
        "marker_size": (6.0, 6.0),
    },
    PAPER_LABEL: {
        "title_font_size": (10, 12),
        "axis_label_font_size": (9, 10),
        "tick_font_size": (8, 9),
        "legend_font_size": (8, 9),
        "annotation_font_size": (8, 99),
        "line_width": (1.5, 1.5),
        "marker_size": (4.0, 4.0),
    },
}


CANONICAL_STYLE = {
    WEB_LABEL: {
        "title_font_size": 18,
        "axis_label_font_size": 15,
        "tick_font_size": 13,
        "legend_font_size": 12,
        "annotation_font_size": 12,
        "line_width": 2.0,
        "marker_size": 6.0,
    },
    PAPER_LABEL: {
        "title_font_size": 11,
        "axis_label_font_size": 9,
        "tick_font_size": 8,
        "legend_font_size": 8,
        "annotation_font_size": 8,
        "line_width": 1.5,
        "marker_size": 4.0,
    },
}


@dataclass
class FigurePayload:
    figure_id: str
    profile: str
    generator_script: str
    generator_args: Sequence[str] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    notes: str | Sequence[str] = ""
    seed: int | None = None
    canvas_px: tuple[int, int] | None = None
    dpi: int | None = None
    style: Mapping[str, Any] | None = None
    git_commit: str | None = None
    python_version: str = ""


def _canonical_style(profile: str) -> dict[str, Any]:
    _ensure_profile(profile)
    return dict(CANONICAL_STYLE[profile])


def _ensure_profile(profile: str) -> str:
    if profile not in (WEB_LABEL, PAPER_LABEL):
        raise ValueError(f"Unsupported profile '{profile}', expected 'web' or 'paper'.")
    return profile


def canonical_canvas_px(profile: str) -> tuple[int, int]:
    _ensure_profile(profile)
    return WEB_CANVAS_PX if profile == WEB_LABEL else PAPER_CANVAS_PX


def canonical_dpi(profile: str) -> int:
    _ensure_profile(profile)
    return WEB_DPI if profile == WEB_LABEL else PAPER_DPI


def canonical_canvas_inches(profile: str) -> tuple[float, float]:
    width_px, height_px = canonical_canvas_px(profile)
    dpi = canonical_dpi(profile)
    return width_px / dpi, height_px / dpi


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _meta_path(path: Path) -> Path:
    return path.parent / "meta" / f"{path.stem}.meta.json"


def _data_path(path: Path) -> Path:
    return path.parent / "data" / f"{path.stem}.npz"


def _normalize_notes(notes: str | Sequence[str]) -> list[str]:
    if isinstance(notes, str):
        lines = [line.strip() for line in notes.splitlines() if line.strip()]
        if not lines:
            return [notes.strip()]
        return lines
    return [str(n).strip() for n in notes if str(n).strip()]


def write_figure_artifacts(
    figure_path: Path,
    *,
    figure_id: str,
    profile: str,
    generator_script: str,
    generator_args: Sequence[str] = (),
    labels: Mapping[str, str],
    units: Mapping[str, str],
    notes: str | Sequence[str],
    seed: int | None = None,
    canvas_px: tuple[int, int] | None = None,
    dpi: int | None = None,
    style: Mapping[str, Any] | None = None,
    payload_extra: Mapping[str, Any] | None = None,
    data_payload: Mapping[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    profile = _ensure_profile(profile)
    can_px = canonical_canvas_px(profile) if canvas_px is None else canvas_px
    can_dpi = canonical_dpi(profile) if dpi is None else dpi

    payload = FigurePayload(
        figure_id=figure_id,
        profile=profile,
        generator_script=generator_script,
        generator_args=tuple(generator_args),
        labels=dict(labels),
        units=dict(units),
        notes=_normalize_notes(notes),
        seed=seed,
        canvas_px=can_px,
        dpi=can_dpi,
        style=dict(style) if style is not None else _canonical_style(profile),
        git_commit=get_git_commit(),
        python_version=__import__("sys").version,
    )
    payload_dict = asdict(payload)
    payload_dict["args"] = tuple(payload_dict.get("generator_args", ()))
    # Keep deterministic output order for CI diffing.
    payload_dict["figure_id"] = figure_id
    payload_dict["seed"] = payload.seed
    payload_dict["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if payload_extra:
        payload_dict["extra"] = dict(payload_extra)

    # Ensure all required keys exist even when optional fields are empty.
    payload_dict = (
        {k: payload_dict.get(k) for k in REQUIRED_META_KEYS}
        | {
            "created_at_utc": payload_dict["created_at_utc"],
            "seed": payload_dict["seed"],
            "python_version": payload_dict["python_version"],
        }
        | ({"extra": payload_dict.get("extra", {})} if "extra" in payload_dict else {})
    )

    target_meta = _meta_path(figure_path)
    target_meta.parent.mkdir(parents=True, exist_ok=True)
    target_meta.write_text(json.dumps(payload_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    data_path: Path | None = None
    if data_payload is not None:
        data_path = _data_path(figure_path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = {}
        for key, value in data_payload.items():
            if isinstance(value, np.ndarray):
                cleaned[key] = value
            else:
                cleaned[key] = np.asarray(value)
        np.savez_compressed(data_path, **cleaned)

    return target_meta, data_path
