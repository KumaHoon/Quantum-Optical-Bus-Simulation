"""Tests for the drift automation simulation."""

from pathlib import Path

from scripts.simulate_24h_drift import generate_drift_profile, run_24h_drift


def test_generate_drift_profile_is_deterministic() -> None:
    first = generate_drift_profile(
        total_hours=2.0,
        steps_per_hour=24,
        seed=11,
        phase_drift_start=0.002,
        phase_drift_end=0.005,
        loss_drift_db_per_hour=0.012,
        squeezing_drift_db_per_hour=-0.04,
    )
    second = generate_drift_profile(
        total_hours=2.0,
        steps_per_hour=24,
        seed=11,
        phase_drift_start=0.002,
        phase_drift_end=0.005,
        loss_drift_db_per_hour=0.012,
        squeezing_drift_db_per_hour=-0.04,
    )

    assert first.time_hours.tolist() == second.time_hours.tolist()
    assert first.phase_rad.tolist() == second.phase_rad.tolist()
    assert first.loss_db.tolist() == second.loss_db.tolist()
    assert first.squeezing_db.tolist() == second.squeezing_db.tolist()


def test_run_24h_drift_writes_artifact(tmp_path: Path) -> None:
    output = tmp_path / "drift_recovery.png"
    produced = run_24h_drift(
        output_path=output,
        total_hours=1.0,
        steps_per_hour=12,
        seed=21,
    )
    assert produced == output
    assert output.exists()
    assert output.stat().st_size > 0
