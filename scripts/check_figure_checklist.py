"""Run the README figure-checklist workflow for reviewer-facing artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from asset_profile import PROFILE_OPTIONS

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / script), *args]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "assets",
        help="Asset root directory.",
    )
    parser.add_argument(
        "--profile",
        default="both",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    parser.add_argument(
        "--target",
        default="advisor",
        choices=("full", "mvp", "advisor"),
        help="Validation boundary. 'advisor' is alias for 'mvp'.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation and only run checklist validation against existing assets.",
    )
    parser.add_argument(
        "--with-roadmap",
        action="store_true",
        help="Generate roadmap/appendix figures after core checklist items.",
    )
    parser.add_argument(
        "--with-style-contract",
        action="store_true",
        help="Run Figure Style Contract validation (metadata/size/profile checks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    target = "mvp" if args.target == "advisor" else args.target

    print(
        f"[INFO] Running figure-checklist workflow. "
        f"output={output_dir} profile={args.profile} target={target}"
    )

    if not args.skip_generate:
        print("[INFO] Core figure generation...")
        _run(
            "generate_dashboard_gallery.py",
            ["--output-dir", str(output_dir), "--profile", args.profile],
        )
        _run(
            "generate_calibration_demo.py",
            ["--output", str(output_dir / "calibration_demo.gif"), "--profile", args.profile],
        )
        _run(
            "generate_control_sweeps.py",
            ["--output-dir", str(output_dir), "--profile", args.profile],
        )

        if args.with_roadmap and target == "full":
            print("[INFO] Optional roadmap generation...")
            _run(
                "generate_advanced_dashboard_gallery.py",
                ["--output-dir", str(output_dir), "--profile", args.profile],
            )
            _run(
                "simulate_24h_drift.py",
                ["--output-dir", str(output_dir), "--profile", args.profile],
            )
            _run(
                "generate_scenario_gallery_gif.py",
                ["--output", str(output_dir / "scenario_gallery.gif"), "--profile", args.profile],
            )
            _run(
                "generate_advanced_gallery_gif.py",
                ["--output", str(output_dir / "advanced_gallery.gif"), "--profile", args.profile],
            )
            _run(
                "generate_advanced_evidence_gif.py",
                ["--output", str(output_dir / "advanced_evidence.gif"), "--profile", args.profile],
            )
            _run(
                "run_gkp_sweep.py",
                ["--output-dir", str(output_dir), "--profile", args.profile],
            )

        print("[INFO] Core validation pass...")
        _run(
            "verify_assets_profiles.py",
            ["--output-dir", str(output_dir), "--profile", args.profile, "--target", target],
        )
        if args.with_style_contract:
            _run(
                "verify_figure_style_contract.py",
                ["--output-dir", str(output_dir), "--profile", args.profile, "--target", target],
            )

    print("[INFO] Figure-checklist artifact audit...")
    _run(
        "README_figure_audit.py",
        ["--output-dir", str(output_dir), "--profile", args.profile, "--target", target],
    )
    if args.with_style_contract:
        _run(
            "verify_figure_style_contract.py",
            ["--output-dir", str(output_dir), "--profile", args.profile, "--target", target],
        )
    print("[OK] Figure-checklist workflow complete.")


if __name__ == "__main__":
    main()
