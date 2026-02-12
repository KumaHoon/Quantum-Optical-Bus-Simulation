"""
Tests for the quantum_optical_bus package.

These tests verify the core physics logic works cross-platform
without requiring Meep or Marimo.
"""

import numpy as np
import pytest

from quantum_optical_bus.hardware import run_hardware_simulation, WaveguideConfig
from quantum_optical_bus.interface import calculate_squeezing


# ── Interface Layer ──────────────────────────────────────────────

class TestCalculateSqueezing:
    def test_zero_power_gives_zero_squeezing(self):
        assert calculate_squeezing(0.0) == 0.0

    def test_positive_power_gives_positive_r(self):
        r = calculate_squeezing(100.0)
        assert r > 0

    def test_squeezing_scales_with_sqrt_power(self):
        r1 = calculate_squeezing(100.0)
        r4 = calculate_squeezing(400.0)
        assert pytest.approx(r4, rel=1e-6) == 2 * r1

    def test_known_value(self):
        # η = 0.1, P = 100  →  r = 0.1 * √100 = 1.0
        assert pytest.approx(calculate_squeezing(100.0), rel=1e-6) == 1.0


# ── Hardware Layer ───────────────────────────────────────────────

class TestHardwareSimulation:
    def test_returns_four_values(self):
        result = run_hardware_simulation()
        assert len(result) == 4

    def test_n_eff_is_positive(self):
        n_eff, _, _, _ = run_hardware_simulation()
        assert n_eff > 0

    def test_mode_area_is_positive(self):
        _, mode_area, _, _ = run_hardware_simulation()
        assert mode_area > 0

    def test_ez_data_shape(self):
        _, _, ez_data, _ = run_hardware_simulation()
        assert ez_data.ndim == 2
        assert ez_data.shape[0] == ez_data.shape[1]

    def test_custom_config(self):
        cfg = WaveguideConfig(core_width=2.0, core_height=1.0)
        n_eff, mode_area, ez_data, extent = run_hardware_simulation(cfg)
        assert n_eff > 0
        assert mode_area > 0


# ── Application Layer (Strawberry Fields) ────────────────────────

class TestQuantumBusModel:
    def test_vacuum_wigner_is_symmetric(self):
        from quantum_optical_bus.application import QuantumBusModel

        model = QuantumBusModel(num_bins=1)
        wigner_data, xvec = model.run_simulation([0.0], [0.0], grid_points=50)

        W, r, theta = wigner_data[0]
        assert r == 0.0
        # Vacuum Wigner should be symmetric: W(x,p) ≈ W(-x,-p)
        assert np.allclose(W, W[::-1, ::-1], atol=1e-6)

    def test_squeezed_state_is_non_vacuum(self):
        from quantum_optical_bus.application import QuantumBusModel

        model = QuantumBusModel(num_bins=1)
        wd_vac, xvec = model.run_simulation([0.0], [0.0], grid_points=50)
        wd_sq, _ = model.run_simulation([100.0], [0.0], grid_points=50)

        W_vac = wd_vac[0][0]
        W_sq = wd_sq[0][0]
        # Squeezed Wigner should differ from vacuum
        assert not np.allclose(W_vac, W_sq, atol=1e-3)
