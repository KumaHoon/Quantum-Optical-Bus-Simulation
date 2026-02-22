"""Audit README-facing figures for figure-checklist review readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image

from asset_profile import PROFILE_OPTIONS, normalize_profiles, resolve_outputs

CORE_MVP_FILES = (
    "dashboard_decoherence.png",
    "calibration_demo.gif",
)
CORE_SWEEP_OPTIONS = (
    "sweep_latency.png",
    "sweep_quantization.png",
)
CORE_FULL_REQUIRED = (
    "dashboard_vacuum.png",
    "dashboard_calibration.png",
    "dashboard_decoherence.png",
    "dashboard_multimode.png",
    "dashboard_topology.png",
    "dashboard_digital_twin.png",
    "sweep_latency.png",
    "sweep_quantization.png",
    "drift_recovery.png",
)
CORE_FULL_GIFS = (
    "calibration_demo.gif",
    "scenario_gallery.gif",
    "advanced_gallery.gif",
    "advanced_evidence.gif",
)


def _target_label(target: str) -> str:
    if target == "advisor":
        return "mvp"
    return target


def _resolve_candidate_paths(base: Path, filename: str, profile: str) -> list[Path]:
    return resolve_outputs(base / filename, profile)


def _assert_file_exists_any(paths: Iterable[Path], description: str) -> bool:
    for path in paths:
        if path.exists():
            return True
        print(f"[WARN] missing: {description} ({path})")
    return False


def _check_png(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def _check_gif(path: Path) -> int:
    with Image.open(path) as im:
        return int(getattr(im, "n_frames", 1))


def _status(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _check_mvp_artifacts(output_dir: Path, profiles: tuple[str, ...], strict: bool) -> int:
    failures = 0
    print("[CHECK] MVP checklist (README-facing)")

    for profile in profiles:
        print(f"- profile: {profile}")

        for name in CORE_MVP_FILES:
            paths = _resolve_candidate_paths(output_dir, name, profile)
            ok = False
            for path in paths:
                if not path.exists():
                    continue
                if name.endswith(".png"):
                    width, height = _check_png(path)
                    if width <= 0 or height <= 0:
                        print(f"  [FAIL] {name} invalid size: {path}")
                        break
                else:
                    frames = _check_gif(path)
                    if frames <= 1:
                        print(f"  [FAIL] {name} must be animated (frames > 1): {path}")
                        break
                    if strict and frames < 3:
                        # keep strict mode lightweight; 2+ frames is still viewable.
                        print(f"  [WARN] {name} has low frame count ({frames}): {path}")
                ok = True
                break

            if not ok:
                failures += 1
            print(f"  {name:28} {_status(ok)}")

        # Require at least one control-sensitivity figure (latency or quantization).
        sweep_ok = False
        for name in CORE_SWEEP_OPTIONS:
            for path in _resolve_candidate_paths(output_dir, name, profile):
                if path.exists():
                    width, height = _check_png(path)
                    if width > 0 and height > 0:
                        sweep_ok = True
                        break
            if sweep_ok:
                break
        if not sweep_ok:
            failures += 1
        print(f"  {'sweep_latency/quantization':28} {_status(sweep_ok)}")

    return failures


def _check_full_artifacts(output_dir: Path, profiles: tuple[str, ...]) -> int:
    failures = 0
    print("[CHECK] Full artifact visibility check")

    for profile in profiles:
        print(f"- profile: {profile}")
        for name in CORE_FULL_REQUIRED:
            paths = _resolve_candidate_paths(output_dir, name, profile)
            ok = _assert_file_exists_any(paths, f"{name}")
            if not ok:
                failures += 1
            print(f"  {name:28} {_status(ok)}")

        for name in CORE_FULL_GIFS:
            paths = _resolve_candidate_paths(output_dir, name, profile)
            ok = False
            for path in paths:
                if not path.exists():
                    continue
                frames = _check_gif(path)
                if frames > 1:
                    ok = True
                    break
            if not ok:
                failures += 1
            print(f"  {name:28} {_status(ok)}")

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
        help="Validation boundary for expected artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict visual checks (warn on low GIF frame count).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = normalize_profiles(args.profile)
    target = _target_label(args.target)
    failures = 0

    if target == "mvp":
        failures = _check_mvp_artifacts(args.output_dir, profiles, strict=args.strict)
    else:
        failures = _check_full_artifacts(args.output_dir, profiles)

    if failures:
        print(f"\nResult: FAIL ({failures} item(s) failed)")
        return 1

    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
