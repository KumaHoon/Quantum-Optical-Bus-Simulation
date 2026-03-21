"""Command-line entrypoints for package workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "src" / "quantum_optical_bus" / "calibration_app.py"


def _run_command(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def _run_script(script_name: str, *args: str) -> None:
    script_path = ROOT_DIR / "scripts" / script_name
    _run_command(sys.executable, str(script_path), *args)


def _normalize_target(target: str) -> str:
    return "mvp" if target == "advisor" else target


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "assets",
        help="Root output directory for generated artifacts.",
    )
    parser.add_argument(
        "--profile",
        default="both",
        choices=("web", "paper", "both"),
        help="Render profile: web, paper, or both.",
    )
    parser.add_argument(
        "--target",
        default="full",
        choices=("full", "mvp", "advisor"),
        help=(
            "'full' validates existing behavior; 'mvp' / 'advisor' limits to review-facing outputs."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run post-generation asset verification.",
    )


def _add_verify_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "assets",
        help="Root output directory for generated artifacts.",
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
        help="'full' validates all contract figures; 'mvp' / 'advisor' validates a subset.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT_DIR / "docs" / "FIGURE_CONTRACT.yaml",
        help="Path to figure contract YAML.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on first failure.",
    )


def _add_app_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port.")
    parser.add_argument("--host", default="localhost", help="Streamlit host.")


def _parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core and optionally advanced artifacts.")
    _add_build_args(parser)
    return parser.parse_args(argv)


def _parse_verify_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated artifacts.")
    _add_verify_args(parser)
    return parser.parse_args(argv)


def _parse_app_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Streamlit dashboard.")
    _add_app_args(parser)
    return parser.parse_args(argv)


def main_build(argv: list[str] | None = None) -> None:
    args = _parse_build_args(argv)
    target = _normalize_target(args.target)
    cmd_args = [
        "--output-dir",
        str(args.output_dir),
        "--profile",
        args.profile,
        "--target",
        target,
    ]
    if args.verify:
        cmd_args.append("--verify")
    _run_script("build_assets_profiles.py", *cmd_args)


def main_verify(argv: list[str] | None = None) -> None:
    args = _parse_verify_args(argv)
    target = _normalize_target(args.target)
    cmd_args = [
        "--output-dir",
        str(args.output_dir),
        "--profile",
        args.profile,
        "--target",
        target,
        "--contract",
        str(args.contract),
    ]
    if args.strict:
        cmd_args.append("--strict")
    _run_script("verify_assets_profiles.py", *cmd_args)


def main_app(argv: list[str] | None = None) -> None:
    args = _parse_app_args(argv)
    _run_command(
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        str(APP_PATH),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantum Optical Bus CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build artifacts.")
    verify_parser = subparsers.add_parser("verify", help="Verify artifacts.")
    app_parser = subparsers.add_parser("app", help="Run Streamlit dashboard.")

    _add_build_args(build_parser)
    _add_verify_args(verify_parser)
    _add_app_args(app_parser)

    args = parser.parse_args(argv)
    if args.command == "build":
        main_build(
            [
                "--output-dir",
                str(args.output_dir),
                "--profile",
                args.profile,
                "--target",
                args.target,
                *(["--verify"] if args.verify else []),
            ]
        )
    elif args.command == "verify":
        main_verify(
            [
                "--output-dir",
                str(args.output_dir),
                "--profile",
                args.profile,
                "--target",
                args.target,
                "--contract",
                str(args.contract),
                *(["--strict"] if args.strict else []),
            ]
        )
    else:
        main_app(["--port", str(args.port), "--host", args.host])


if __name__ == "__main__":
    main()
