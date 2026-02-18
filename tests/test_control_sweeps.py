"""Tests for control-constraints sweep generation."""

from pathlib import Path

from quantum_optical_bus.control import simulate_phase_drift

from scripts.generate_control_sweeps import (
    generate_control_sweeps,
    run_latency_sweep,
    run_quantization_sweep,
)


def test_latency_sweep_is_deterministic_and_increasing() -> None:
    """Zero-latency should outperform higher-latency settings."""
    phase = simulate_phase_drift(T=120, step_sigma=0.0, drift_rate=0.01, seed=3)
    results = run_latency_sweep(
        true_phase=phase,
        latencies=[0, 1, 2, 3],
        measurement_sigma=0.0,
        seed=5,
    )

    rms_values = [r.rms_residual_phase_error for r in results]
    assert rms_values[0] <= rms_values[-1] + 1e-12
    assert results[0].key == 0
    assert results[-1].key == 3


def test_quantization_sweep_improves_with_more_bits() -> None:
    """Higher bit width should not worsen zero-noise residual in this model."""
    phase = simulate_phase_drift(T=200, step_sigma=0.0, drift_rate=0.015, seed=4)
    results = run_quantization_sweep(
        true_phase=phase,
        bit_widths=[2, 3, 4, 6, 8],
        measurement_sigma=0.0,
        seed=7,
    )

    rms_values = [r.rms_residual_phase_error for r in results]
    assert rms_values[0] >= rms_values[-1] - 1e-12
    assert [r.key for r in results] == [2, 3, 4, 6, 8]


def test_generate_control_sweeps_writes_expected_assets(tmp_path: Path) -> None:
    paths = generate_control_sweeps(output_dir=tmp_path, seed=9, n_steps=140)

    assert paths["sweep_latency"].name == "sweep_latency.png"
    assert paths["sweep_quantization"].name == "sweep_quantization.png"
    assert paths["sweep_latency"].exists()
    assert paths["sweep_quantization"].exists()
