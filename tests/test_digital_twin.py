"""Tests for digital twin estimation and latency-aware control."""

import numpy as np

from quantum_optical_bus.control import apply_feedback_with_latency, simulate_phase_drift
from quantum_optical_bus.estimation import fit_eta_and_loss


class TestParameterEstimation:
    def test_fit_recovers_eta_and_loss_on_synthetic_data(self):
        rng = np.random.default_rng(123)
        eta_true = 0.12
        loss_true_db = 2.0

        powers = np.linspace(5.0, 180.0, 60)
        r = eta_true * np.sqrt(powers)
        transmissivity = 10.0 ** (-loss_true_db / 10.0)
        vac = 0.5
        var_x = transmissivity * (vac * np.exp(-2.0 * r)) + (1.0 - transmissivity) * vac
        var_p = transmissivity * (vac * np.exp(2.0 * r)) + (1.0 - transmissivity) * vac

        data = {
            "timestamp": np.arange(powers.size, dtype=float),
            "pump_power_mw": powers,
            "measured_var_x": var_x + rng.normal(0.0, 0.002, size=powers.size),
            "measured_var_p": var_p + rng.normal(0.0, 0.008, size=powers.size),
            "estimated_loss_db": np.full(powers.size, 1.0, dtype=float),
        }

        eta_hat, loss_hat, diagnostics = fit_eta_and_loss(data)

        assert diagnostics["success"]
        assert abs(eta_hat - eta_true) < 0.015
        assert abs(loss_hat - loss_true_db) < 0.45


class TestLatencyControl:
    def test_increasing_latency_worsens_residual_error(self):
        # Deterministic drift path avoids random non-monotonic artifacts.
        true_phase = simulate_phase_drift(T=80, step_sigma=0.0, drift_rate=0.02, seed=0)
        latencies = [0, 1, 3, 6]

        rms_errors = []
        for lat in latencies:
            result = apply_feedback_with_latency(
                latency_steps=lat,
                true_phase=true_phase,
                measurement_sigma=0.0,
                seed=1,
            )
            rms_errors.append(result["rms_residual_phase_error"])

        for idx in range(len(rms_errors) - 1):
            assert rms_errors[idx] <= rms_errors[idx + 1] + 1e-12, (
                f"Latency trend violated at {latencies[idx]}->{latencies[idx+1]}: "
                f"{rms_errors[idx]} vs {rms_errors[idx+1]}"
            )
