"""Fit intrinsic loss/squeezing model from lab calibration CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_optical_bus.estimation import fit_eta_and_loss


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit (eta, loss_db) from calibration CSV in the project data schema "
            "and print a compact report."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/calibration_sample.csv"),
        help="CSV file containing calibration columns (default: data/raw/calibration_sample.csv)",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=("auto", "variance", "squeezing_db"),
        help="Estimator model to use.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Optional JSON path to write a compact fit summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    eta_hat, loss_db_hat, diagnostics = fit_eta_and_loss(args.data, model=args.model)

    report = {
        "data": str(args.data),
        "model": diagnostics.get("model", args.model),
        "eta_hat": eta_hat,
        "loss_db_hat": loss_db_hat,
        "diagnostics": diagnostics,
    }

    print("=== lab calibration fit ===")
    print(f"data: {report['data']}")
    print(f"model: {report['model']}")
    print(f"eta_hat: {eta_hat:.6f}")
    print(f"loss_db_hat: {loss_db_hat:.6f} dB")
    print(f"fit rmse: {diagnostics.get('rmse', float('nan')):.6e}")
    print(f"n_samples: {diagnostics.get('n_samples', 'NA')}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote fit summary: {args.out_json}")


if __name__ == "__main__":
    main()
