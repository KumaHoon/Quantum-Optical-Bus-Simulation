"""End-to-end onboarding flow for raw calibration CSV data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from asset_profile import PROFILE_OPTIONS

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run_script(script_name: str, args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / script_name), *args]
    subprocess.run(cmd, check=True)


def _collect_inputs(paths: list[Path]) -> list[Path]:
    inputs: list[Path] = []
    for p in paths:
        if p.is_dir():
            inputs.extend(sorted(p.glob("*.csv")))
        elif p.suffix.lower() == ".csv":
            inputs.append(p)
        else:
            raise ValueError(f"Unsupported input path for onboarding: {p}")

    if not inputs:
        raise FileNotFoundError(
            "No CSV files found for onboarding. Provide data/*.csv in --data-path."
        )

    return inputs


def _append_history(history_path: Path, records: list[dict]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the raw-data onboarding flow: fit calibration data, regenerate "
            "assets, and optionally verify outputs."
        )
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        type=Path,
        default=[Path("data/raw")],
        help="One or more CSV files or directories containing CSV exports.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=("auto", "variance", "squeezing_db"),
        help="Estimator model passed to fit_lab_data.",
    )
    parser.add_argument(
        "--profile",
        default="both",
        choices=PROFILE_OPTIONS,
        help="Asset output profile.",
    )
    parser.add_argument(
        "--target",
        default="advisor",
        choices=("full", "mvp", "advisor"),
        help="Build target to use for asset generation.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run asset verification after generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "assets",
        help="Output root for generated assets.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=ROOT_DIR / "data" / "onboarding_history.jsonl",
        help="Append onboarding diagnostics to this file.",
    )
    parser.add_argument(
        "--fit-summary-dir",
        type=Path,
        default=ROOT_DIR / "data" / "fit_summaries",
        help="Directory for per-file fit summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_files = _collect_inputs([Path(p) for p in args.data_path])
    fit_summary_dir = args.fit_summary_dir
    fit_summary_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for file in data_files:
        out_json = fit_summary_dir / f"{file.stem}_fit.json"
        _run_script(
            "fit_lab_data.py",
            [
                "--data",
                str(file),
                "--model",
                args.model,
                "--out-json",
                str(out_json),
            ],
        )
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        payload["source_file"] = str(file)
        payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        records.append(payload)

    if args.output_dir:
        _run_script(
            "build_assets_profiles.py",
            [
                "--output-dir",
                str(args.output_dir),
                "--profile",
                args.profile,
                "--target",
                args.target,
            ],
        )

    if args.verify:
        _run_script(
            "verify_assets_profiles.py",
            [
                "--output-dir",
                str(args.output_dir),
                "--profile",
                args.profile,
                "--target",
                args.target,
            ],
        )

    _append_history(args.history, records)

    for record in records:
        print(
            f"[OK] fit: {Path(record['data']).name} "
            f"model={record['model']} eta={record['eta_hat']:.6f} "
            f"loss_db={record['loss_db_hat']:.6f} rmse={record['diagnostics'].get('rmse', float('nan')):.6e}"
        )

    print(
        f"[OK] onboarding complete. target={args.target}, profile={args.profile}, "
        f"files={len(records)}"
    )


if __name__ == "__main__":
    main()
