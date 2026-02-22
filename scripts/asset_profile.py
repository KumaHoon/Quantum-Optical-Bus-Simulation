"""Shared profile/output helpers for web/paper artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROFILE_OPTIONS = ("web", "paper", "both")


def normalize_profiles(profile: str) -> tuple[str, ...]:
    """Return one or both rendering profiles as a tuple."""
    if profile == "both":
        return ("web", "paper")
    if profile not in PROFILE_OPTIONS:
        raise ValueError(f"Unsupported profile '{profile}'. Choose from {PROFILE_OPTIONS}.")
    return (profile,)


def resolve_outputs(output: Path, profile: str) -> list[Path]:
    """Resolve output path(s) for web/paper/both under a single asset root.

    If output.parent is already ``web`` or ``paper``, avoid duplicating that segment.
    """
    output_path = Path(output)
    base = output_path.parent
    name = output_path.name
    outputs: list[Path] = []

    profiles = normalize_profiles(profile)
    if "web" in profiles:
        web_base = base if base.name == "web" else base / "web"
        outputs.append(web_base / name)
    if "paper" in profiles:
        paper_base = base if base.name == "paper" else base / "paper"
        outputs.append(paper_base / name)

    # remove duplicates while preserving order
    return list(dict.fromkeys(outputs))


def iter_profile_outputs(base_dir: Path, filename: str, profile: str) -> list[Path]:
    """Resolve a (base_dir, filename) pair for compatibility with existing callers."""
    return resolve_outputs(base_dir / filename, profile)


def candidate_output_paths(base_dir: Path, filename: str) -> Iterable[Path]:
    """Yield source-image candidates in root/web/paper order."""
    base = Path(base_dir)
    yield base / filename
    yield base / "web" / filename
    yield base / "paper" / filename
