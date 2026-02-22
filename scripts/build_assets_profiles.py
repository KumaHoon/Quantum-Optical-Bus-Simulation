"""Generate all web/paper outputs from a single command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from asset_profile import PROFILE_OPTIONS
except ModuleNotFoundError:
    from scripts.asset_profile import PROFILE_OPTIONS

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"


def run_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / script), *args]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Root output directory for generated artifacts.",
    )
    parser.add_argument(
        "--profile",
        default="both",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run post-generation asset validation.",
    )
    parser.add_argument(
        "--target",
        default="full",
        choices=("full", "mvp", "advisor"),
        help=(
            "Generation target. "
            "'full' keeps existing behavior; 'mvp' (alias 'advisor') generates "
            "a reviewer-sized subset for squeezed-light KPI review."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    profile = args.profile
    target = "mvp" if args.target == "advisor" else args.target

    print("Generating web/paper core artifacts ...")

    run_script(
        "generate_dashboard_gallery.py",
        ["--output-dir", str(output_dir), "--profile", profile],
    )
    run_script(
        "generate_calibration_demo.py",
        ["--output", str(output_dir / "calibration_demo.gif"), "--profile", profile],
    )
    run_script(
        "generate_control_sweeps.py",
        ["--output-dir", str(output_dir), "--profile", profile],
    )

    if target == "full":
        print("Generating roadmap/advanced artifacts ...")
        run_script(
            "generate_advanced_dashboard_gallery.py",
            ["--output-dir", str(output_dir), "--profile", profile],
        )
        run_script(
            "simulate_24h_drift.py",
            ["--output-dir", str(output_dir), "--profile", profile],
        )

    if target == "full":
        run_script(
            "generate_scenario_gallery_gif.py",
            ["--output", str(output_dir / "scenario_gallery.gif"), "--profile", profile],
        )
        run_script(
            "generate_advanced_gallery_gif.py",
            ["--output", str(output_dir / "advanced_gallery.gif"), "--profile", profile],
        )
        run_script(
            "generate_advanced_evidence_gif.py",
            ["--output", str(output_dir / "advanced_evidence.gif"), "--profile", profile],
        )
        run_script(
            "run_gkp_sweep.py",
            ["--output-dir", str(output_dir), "--profile", profile],
        )

    if args.verify:
        run_script(
            "verify_assets_profiles.py",
            ["--output-dir", str(output_dir), "--profile", profile, "--target", target],
        )

    print(f"[OK] Build complete. Profile={profile}, output-dir={output_dir}, target={target}")


if __name__ == "__main__":
    main()
