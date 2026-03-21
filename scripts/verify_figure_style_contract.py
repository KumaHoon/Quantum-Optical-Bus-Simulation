"""Verify figure outputs against the repository style contract."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

from asset_profile import PROFILE_OPTIONS, normalize_profiles
from figure_style_contract import canonical_canvas_px, canonical_dpi


CORE_WEB_MVP = (
    "dashboard_vacuum.png",
    "dashboard_calibration.png",
    "dashboard_decoherence.png",
)
SWEEP_FILES = ("sweep_latency.png", "sweep_quantization.png")
CORE_GIFS = ("calibration_demo.gif",)


OPTIONAL_FULL_PNGS = (
    "dashboard_vacuum.png",
    "dashboard_calibration.png",
    "dashboard_decoherence.png",
    "dashboard_multimode.png",
    "dashboard_topology.png",
    "dashboard_digital_twin.png",
    "sweep_latency.png",
    "sweep_quantization.png",
    "drift_recovery.png",
    "gkp_proxy.png",
)
OPTIONAL_FULL_GIFS = (
    "scenario_gallery.gif",
    "advanced_gallery.gif",
    "advanced_evidence.gif",
)


def _resolve_candidate_paths(base: Path, filename: str, profile: str) -> list[Path]:
    from asset_profile import resolve_outputs

    return resolve_outputs(base / filename, profile)


def _read_png_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _read_gif_frames(path: Path) -> int:
    with Image.open(path) as im:
        return int(getattr(im, "n_frames", 0))


def _read_meta(path: Path) -> dict:
    meta_path = path.parent / "meta" / f"{path.stem}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    return __import__("json").loads(meta_path.read_text(encoding="utf-8"))


def _validate_axis_units(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    return bool(
        re.search(r"\[[^\]]+\]$", text)
        or re.search(r"\([^)]+\)$", text)
        or "unitless" in text.lower()
    )


def _check_required_png(path: Path, profile: str, strict_units: bool) -> int:
    width, height = _read_png_size(path)
    expected_w, expected_h = canonical_canvas_px(profile)
    if (width, height) != (expected_w, expected_h):
        raise AssertionError(
            f"Size mismatch for {path}: {(width, height)} != {(expected_w, expected_h)}"
        )

    meta = _read_meta(path)
    if int(meta.get("dpi", -1)) != canonical_dpi(profile):
        raise AssertionError(
            f"DPI mismatch for {path}: {meta.get('dpi')} != {canonical_dpi(profile)}"
        )
    if tuple(meta.get("canvas_px", [])) != tuple(canonical_canvas_px(profile)):
        raise AssertionError(
            f"Metadata canvas mismatch for {path}: {meta.get('canvas_px')} != {canonical_canvas_px(profile)}"
        )
    if meta.get("profile") != profile:
        raise AssertionError(f"Profile mismatch for {path}: {meta.get('profile')} != {profile}")

    labels = meta.get("labels") or {}
    if not labels.get("title"):
        raise AssertionError(f"Missing title label in metadata: {path}")
    if not labels.get("xlabel"):
        raise AssertionError(f"Missing x-axis label in metadata: {path}")
    if not labels.get("ylabel"):
        raise AssertionError(f"Missing y-axis label in metadata: {path}")
    if strict_units and not _validate_axis_units(labels["xlabel"]):
        raise AssertionError(f"X-axis label lacks unit: {path}: {labels['xlabel']}")
    if strict_units and not _validate_axis_units(labels["ylabel"]):
        raise AssertionError(f"Y-axis label lacks unit: {path}: {labels['ylabel']}")

    if not meta.get("notes"):
        raise AssertionError(f"Missing notes in metadata: {path}")
    if not isinstance(meta["notes"], list) or not any(str(line).strip() for line in meta["notes"]):
        raise AssertionError(f"Invalid notes format in metadata: {path}")

    if not isinstance(meta.get("labels"), dict):
        raise AssertionError(f"Invalid labels object in metadata: {path}")

    return 0


def _check_gif(path: Path, profile: str) -> int:
    frames = _read_gif_frames(path)
    if frames <= 1:
        raise AssertionError(f"GIF must contain >1 frame: {path}")
    # Metadata is still required for agent-driven reproducibility.
    meta = _read_meta(path)
    if meta.get("profile") != profile:
        raise AssertionError(f"Profile mismatch for {path}: {meta.get('profile')} != {profile}")
    return 0


def _check_figure_set(output_root: Path, profile: str, target: str) -> int:
    failures = 0
    strict_units = True

    if target == "mvp":
        # MVP checks the 3-panel baseline + one sensitivity route + calibration demo.
        for name in CORE_WEB_MVP:
            paths = _resolve_candidate_paths(output_root, name, profile)
            if not paths:
                raise FileNotFoundError(f"No path candidate for {name} ({profile})")
            checked = False
            for path in paths:
                if path.exists():
                    _check_required_png(path, profile=profile, strict_units=strict_units)
                    if profile == "paper":
                        pdf_path = path.with_suffix(".pdf")
                        if not pdf_path.exists():
                            raise FileNotFoundError(f"Missing paper PDF companion: {pdf_path}")
                    checked = True
                    break
            if not checked:
                raise FileNotFoundError(f"Missing required file {name} for profile {profile}")

        ok_sweep = False
        for name in SWEEP_FILES:
            paths = _resolve_candidate_paths(output_root, name, profile)
            for path in paths:
                if path.exists():
                    _check_required_png(path, profile=profile, strict_units=strict_units)
                    if profile == "paper":
                        pdf_path = path.with_suffix(".pdf")
                        if not pdf_path.exists():
                            raise FileNotFoundError(f"Missing paper PDF companion: {pdf_path}")
                    ok_sweep = True
                    break
            if ok_sweep:
                break
        if not ok_sweep:
            raise FileNotFoundError(
                f"Missing sensitivity sweep for profile {profile}: expected one of {SWEEP_FILES}"
            )

        for name in CORE_GIFS:
            paths = _resolve_candidate_paths(output_root, name, profile)
            checked = False
            for path in paths:
                if path.exists():
                    _check_gif(path, profile=profile)
                    checked = True
                    break
            if not checked:
                raise FileNotFoundError(f"Missing required animation {name} for profile {profile}")

        return failures

    # full: verify all required figures + roadmap-only roadmap-aware outputs where present
    # Full/advisor mode: run contract checks on all baseline + roadmap figures when present.
    # Missing roadmap artifacts are warning-level only and do not fail baseline acceptance.
    for name in OPTIONAL_FULL_PNGS:
        paths = _resolve_candidate_paths(output_root, name, profile)
        checked = False
        for path in paths:
            if path.exists():
                _check_required_png(path, profile=profile, strict_units=strict_units)
                if profile == "paper":
                    pdf_path = path.with_suffix(".pdf")
                    if not pdf_path.exists():
                        raise FileNotFoundError(f"Missing paper PDF companion: {pdf_path}")
                checked = True
                break

        if not checked:
            print(f"[WARN] Optional full-route figure missing: {name} ({profile})")

    for name in OPTIONAL_FULL_GIFS:
        paths = _resolve_candidate_paths(output_root, name, profile)
        checked = False
        for path in paths:
            if path.exists():
                _check_gif(path, profile=profile)
                checked = True
                break
        if not checked:
            # Roadmap figures are optional unless explicitly generated by full workflow.
            print(f"[WARN] Optional full-route animation missing: {name} ({profile})")

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets",
        help="Asset root directory.",
    )
    parser.add_argument(
        "--profile",
        default="both",
        choices=PROFILE_OPTIONS,
        help="web, paper, or both.",
    )
    parser.add_argument(
        "--target",
        default="advisor",
        choices=("full", "mvp", "advisor"),
        help="Validation boundary. 'advisor' is an alias for 'mvp'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = normalize_profiles(args.profile)
    target = "mvp" if args.target == "advisor" else args.target

    for profile in profiles:
        _check_figure_set(args.output_dir, profile, target)
        print(f"[OK] style contract check passed for {profile} ({target})")

    print("[OK] Figure Style Contract validation passed.")


if __name__ == "__main__":
    main()
