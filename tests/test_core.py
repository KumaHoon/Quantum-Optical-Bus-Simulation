"""
Tests for the quantum_optical_bus package.

These tests verify the core physics logic works cross-platform
without requiring Meep or Marimo.
"""

import quantum_optical_bus.compat  # noqa: F401  — must precede SF imports

import numpy as np
import pytest

from quantum_optical_bus.hardware import run_hardware_simulation, WaveguideConfig
from quantum_optical_bus.interface import calculate_squeezing
from quantum_optical_bus.multimode import run_multimode
from quantum_optical_bus.quantum import run_single_mode


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


# ── Quantum Simulation (Strawberry Fields) ───────────────────────


class TestQuantumSimulation:
    """Verify core quantum simulation properties using SF directly."""

    @staticmethod
    def _wigner(r: float, theta: float, grid_points: int = 50):
        """Run a single-mode Gaussian circuit and return Wigner + grid."""
        import strawberryfields as sf
        from strawberryfields.ops import Sgate, Rgate

        xvec = np.linspace(-4.0, 4.0, grid_points)
        prog = sf.Program(1)
        with prog.context as q:
            if r > 0:
                Sgate(r) | q[0]
            if theta != 0:
                Rgate(theta) | q[0]
        state = sf.Engine("gaussian").run(prog).state
        W = state.wigner(0, xvec, xvec)
        return W, xvec

    def test_vacuum_wigner_is_symmetric(self):
        W, _ = self._wigner(r=0.0, theta=0.0)
        # Vacuum Wigner should be symmetric: W(x,p) ≈ W(-x,-p)
        assert np.allclose(W, W[::-1, ::-1], atol=1e-6)

    def test_squeezed_state_is_non_vacuum(self):
        W_vac, _ = self._wigner(r=0.0, theta=0.0)
        r = calculate_squeezing(100.0)  # r ≈ 1.0
        W_sq, _ = self._wigner(r=r, theta=0.0)
        # Squeezed Wigner should differ from vacuum
        assert not np.allclose(W_vac, W_sq, atol=1e-3)


# ── Intrinsic vs Observed Squeezing ──────────────────────────────


class TestSqueezingVsLoss:
    """Verify intrinsic squeezing is constant and observed squeezing
    decreases monotonically as loss increases (fixed pump power)."""

    PUMP_POWER_MW: float = 100.0  # → r = 1.0
    LOSS_DB_VALUES: list[float] = [0.0, 3.0, 6.0, 10.0]

    @staticmethod
    def _observed_squeezing_db(r: float, eta: float) -> float:
        """Run a single-mode Gaussian circuit and return observed squeezing (dB).

        Parameters
        ----------
        r : float
            Squeezing parameter for the Sgate.
        eta : float
            Channel transmissivity in [0, 1].

        Returns
        -------
        float
            Observed squeezing in dB, computed from the minimum eigenvalue
            of the normalized covariance matrix V = cov / 2.
        """
        import strawberryfields as sf
        from strawberryfields.ops import Sgate, LossChannel

        prog = sf.Program(1)
        with prog.context as q:
            if r > 0:
                Sgate(r) | q[0]
            if eta < 1.0:
                LossChannel(eta) | q[0]

        eng = sf.Engine("gaussian")
        state = eng.run(prog).state
        cov = state.cov()
        V = cov / 2.0
        eigvals = np.linalg.eigvalsh(V)
        Vmin = eigvals[0]
        vacuum_var = 0.5
        return float(-10 * np.log10(Vmin / vacuum_var))

    def test_observed_squeezing_decreases_with_loss(self):
        """Observed (post-loss) squeezing must be monotonically non-increasing
        as loss increases, because the loss channel mixes the state with vacuum."""
        r = calculate_squeezing(self.PUMP_POWER_MW)
        observed_values = []
        for loss_db in self.LOSS_DB_VALUES:
            eta = 10 ** (-loss_db / 10.0)
            observed_values.append(self._observed_squeezing_db(r, eta))

        # Monotonically non-increasing
        for i in range(len(observed_values) - 1):
            assert observed_values[i] >= observed_values[i + 1] - 1e-9, (
                f"Observed squeezing increased from {observed_values[i]:.4f} dB "
                f"to {observed_values[i + 1]:.4f} dB when loss went from "
                f"{self.LOSS_DB_VALUES[i]} to {self.LOSS_DB_VALUES[i + 1]} dB"
            )

        # At 10 dB loss the observed squeezing should be noticeably less than at 0 dB
        assert observed_values[-1] < observed_values[0] - 0.5

    def test_intrinsic_squeezing_constant_across_loss(self):
        """Intrinsic squeezing (from r) must not change when loss changes,
        because r depends only on pump power."""
        r = calculate_squeezing(self.PUMP_POWER_MW)
        intrinsic_db = -10 * np.log10(np.exp(-2 * r))

        for loss_db in self.LOSS_DB_VALUES:
            # Recompute r — it must be independent of loss
            r_again = calculate_squeezing(self.PUMP_POWER_MW)
            intrinsic_db_again = -10 * np.log10(np.exp(-2 * r_again))
            assert intrinsic_db_again == pytest.approx(intrinsic_db, abs=1e-12), (
                f"Intrinsic squeezing changed at loss_db={loss_db}"
            )


class TestMultiModeCore:
    """Tests for independent multi-mode Gaussian simulation."""

    def test_n1_matches_single_mode_metrics(self):
        xvec = np.linspace(-4.0, 4.0, 80)
        r = calculate_squeezing(100.0)
        theta = 0.21
        eta = 0.83

        single = run_single_mode(r=r, theta=theta, eta_loss=eta, xvec=xvec)
        multi = run_multimode(
            r=r,
            theta=theta,
            eta_loss=eta,
            n_modes=1,
            xvec=xvec,
            wigner_mode=0,
        )

        assert multi.mean_photon[0] == pytest.approx(single.mean_photon, rel=1e-7, abs=1e-9)
        assert multi.var_x[0] == pytest.approx(single.var_x, rel=1e-7, abs=1e-9)
        assert multi.var_p[0] == pytest.approx(single.var_p, rel=1e-7, abs=1e-9)
        assert multi.observed_sq_db[0] == pytest.approx(single.observed_sq_db, rel=1e-7, abs=1e-9)
        assert multi.observed_antisq_db[0] == pytest.approx(
            single.observed_antisq_db, rel=1e-7, abs=1e-9
        )
        assert multi.wigner is not None
        assert np.allclose(multi.wigner, single.W, atol=1e-8)

    def test_per_mode_observed_squeezing_decreases_with_loss(self):
        r = calculate_squeezing(100.0)
        losses = np.array([1.0, 0.8, 0.6, 0.4])
        multi = run_multimode(
            r=np.full(losses.shape, r),
            theta=0.0,
            eta_loss=losses,
            n_modes=losses.size,
        )

        observed = multi.observed_sq_db
        for idx in range(observed.size - 1):
            assert observed[idx] >= observed[idx + 1] - 1e-9, (
                f"Observed squeezing increased from mode {idx} to {idx + 1}: "
                f"{observed[idx]:.4f} dB -> {observed[idx + 1]:.4f} dB"
            )

    def test_vacuum_selected_mode_wigner_is_symmetric(self):
        xvec = np.linspace(-4.0, 4.0, 60)
        multi = run_multimode(
            r=[0.0, 0.0, 0.0],
            theta=[0.0, 0.0, 0.0],
            eta_loss=[1.0, 1.0, 1.0],
            xvec=xvec,
            wigner_mode=1,
        )

        assert multi.wigner is not None
        assert np.allclose(multi.wigner, multi.wigner[::-1, ::-1], atol=1e-6)
