"""Verify generated assets against FIGURE_CONTRACT.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from figure_verification import verify_profile
else:
    from scripts.figure_verification import verify_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets",
        help="Root directory containing generated artifacts.",
    )
    parser.add_argument(
        "--profile",
        default="both",
        choices=("web", "paper", "both"),
        help="Validate web, paper, or both outputs.",
    )
    parser.add_argument(
        "--target",
        default="full",
        choices=("full", "mvp", "advisor"),
        help=(
            "Validation target. "
            "'full' validates all contract figures, "
            "'mvp' / 'advisor' validates README-facing subset."
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "FIGURE_CONTRACT.yaml",
        help="Path to figure contract YAML.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on first failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = "mvp" if args.target == "advisor" else args.target

    profiles = (args.profile,) if args.profile != "both" else ("web", "paper")
    all_failures = False
    for profile in profiles:
        report = verify_profile(
            profile,
            target,
            contract_path=args.contract,
            output_root=args.output_dir,
            strict=args.strict,
        )
        for result in report.results:
            if result.ok:
                print(f"[OK] {profile}: {result.figure_id} {result.validator}")
            else:
                print(result.format())
                all_failures = True
                if args.strict:
                    raise SystemExit(1)
        if not report.ok and not report.results:
            all_failures = True
        if report.ok:
            print(f"[OK] {profile}: all checks passed ({report.summary.get('passed', 0)} ok, {report.summary.get('failed', 0)} failed)")

    if all_failures:
        raise SystemExit(1)
    print("[OK] Asset verification passed.")


if __name__ == "__main__":
    main()
