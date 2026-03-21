"""
Bridge figure-contract checks into pytest-level tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.figure_verification import (
    ValidationResult,
    loss_monotonic_observed,
    verify_figure,
)


def _minimal_meta_payload() -> dict:
    return {
        "figure_id": "decoherence_contract_check",
        "profile": "web",
        "created_at_utc": "2026-02-22T00:00:00Z",
        "generator_script": "tests/test_figure_contract_validation.py",
        "labels": {"x": "Loss [dB]", "y": "Observed squeezing [dB]"},
        "units": {"x": "Loss [dB]", "y": "Squeezing [dB]"},
        "canvas_px": [1600, 1000],
        "dpi": 150,
    }


def test_loss_monotonic_observed_validator_accepts_physical_curve() -> None:
    npz = {
        "loss_db": np.array([0.0, 3.0, 1.0]),
        "observed_sq_db_x": np.array([8.0, 6.8, 7.2]),
    }
    results = loss_monotonic_observed(
        "dashboard_decoherence",
        "web",
        npz,
        loss_key="loss_db",
        observed_sq_key="observed_sq_db_x",
    )
    assert all(isinstance(item, ValidationResult) for item in results)
    assert all(item.ok for item in results)


def test_loss_monotonic_observed_validator_flags_increase_in_abs_squeezing() -> None:
    npz = {
        "loss_db": np.array([0.0, 1.0, 2.0]),
        "observed_sq_db_x": np.array([8.0, 8.05, 7.8]),
    }
    results = loss_monotonic_observed(
        "dashboard_decoherence",
        "web",
        npz,
        loss_key="loss_db",
        observed_sq_key="observed_sq_db_x",
        tol_abs=0.0,
    )
    assert len(results) == 1
    assert not results[0].ok
    assert results[0].validator == "loss_monotonic_observed"


def test_verify_figure_reuses_contract_monotonic_validation(tmp_path: Path) -> None:
    png = tmp_path / "dashboard_decoherence.png"
    png.write_bytes(b"")
    npz = tmp_path / "dashboard_decoherence.npz"
    np.savez(
        npz,
        loss_db=np.array([0.0, 3.0, 1.0]),
        observed_sq_db_x=np.array([8.0, 6.8, 7.2]),
    )
    meta = tmp_path / "dashboard_decoherence.meta.json"
    meta.write_text(
        json.dumps(_minimal_meta_payload()),
        encoding="utf-8",
    )

    contract = {
        "figures": [
            {
                "id": "dashboard_decoherence",
                "outputs": {"web_png": str(png)},
                "companions": {
                    "web_meta": str(meta),
                    "web_npz": str(npz),
                },
                "validations": [
                    {
                        "type": "loss_monotonic_observed",
                        "loss_key": "loss_db",
                        "observed_sq_key": "observed_sq_db_x",
                        "direction": "toward_zero",
                    }
                ],
            }
        ]
    }

    results = verify_figure("dashboard_decoherence", "web", contract=contract)
    assert all(item.ok for item in results)
