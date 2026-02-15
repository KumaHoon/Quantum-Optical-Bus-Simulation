"""End-to-end digital twin demo: synthetic fit + latency control report."""

from __future__ import annotations

import numpy as np

from quantum_optical_bus.control import apply_feedback_with_latency, simulate_phase_drift
from quantum_optical_bus.estimation import fit_eta_and_loss


def main() -> None:
    rng = np.random.default_rng(7)

    # Ground truth for synthetic calibration data.
    eta_true = 0.115
    loss_true_db = 1.7
    powers = np.linspace(5.0, 150.0, 40)

    r = eta_true * np.sqrt(powers)
    transmissivity = 10.0 ** (-loss_true_db / 10.0)
    vac = 0.5
    var_x = transmissivity * (vac * np.exp(-2.0 * r)) + (1.0 - transmissivity) * vac
    var_p = transmissivity * (vac * np.exp(2.0 * r)) + (1.0 - transmissivity) * vac

    noisy_data = {
        "timestamp": np.arange(powers.size, dtype=float),
        "pump_power_mw": powers,
        "measured_var_x": var_x + rng.normal(0.0, 0.003, size=powers.size),
        "measured_var_p": var_p + rng.normal(0.0, 0.010, size=powers.size),
        "estimated_loss_db": np.full(powers.size, 1.0, dtype=float),
    }

    eta_hat, loss_hat, diag = fit_eta_and_loss(noisy_data, model="variance")

    # Drift + latency control run.
    phase = simulate_phase_drift(T=400, step_sigma=0.015, drift_rate=0.002, seed=11)
    latency_levels = [0, 2, 6]
    control_results = [
        apply_feedback_with_latency(
            latency_steps=lat,
            true_phase=phase,
            measurement_sigma=0.002,
            seed=17,
        )
        for lat in latency_levels
    ]

    print("=== Digital Twin Demo Report ===")
    print(f"Ground truth: eta={eta_true:.4f}, loss_db={loss_true_db:.4f}")
    print(f"Estimated   : eta={eta_hat:.4f}, loss_db={loss_hat:.4f}")
    print(f"Fit RMSE    : {diag['rmse']:.6f} ({diag['model']})")
    print("")
    print("Latency impact on residual phase error:")
    for lat, res in zip(latency_levels, control_results):
        print(
            f"  latency={lat:2d} | rms_error={res['rms_residual_phase_error']:.6f} "
            f"| mean_retention={res['mean_retention_proxy']:.6f}"
        )


if __name__ == "__main__":
    main()
