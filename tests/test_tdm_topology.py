"""Tests for topology-driven TDM simulation."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantum_optical_bus.multimode import run_multimode
from quantum_optical_bus.tdm_topology import load_topology_config, simulate_topology


class TestTopologyConfig:
    def test_json_config_parsing(self, tmp_path: Path):
        config = {
            "n_modes": 3,
            "squeezing_r": [0.2, 0.3, 0.4],
            "squeezing_phi": 0.0,
            "phase_shifts": [0.0, 0.1, 0.2],
            "loss": [1.0, 0.9, 0.8],
            "couplings": [
                {"i": 0, "j": 1, "theta": 0.25, "phi": 0.0, "eta_loss": 0.98},
                {"i": 1, "j": 2, "theta": 0.20},
            ],
        }
        path = tmp_path / "topology.json"
        path.write_text(json.dumps(config), encoding="utf-8")

        parsed = load_topology_config(path)
        assert parsed.n_modes == 3
        assert np.allclose(parsed.squeezing_r, [0.2, 0.3, 0.4])
        assert np.allclose(parsed.phase_shifts, [0.0, 0.1, 0.2])
        assert np.allclose(parsed.loss, [1.0, 0.9, 0.8])
        assert len(parsed.couplings) == 2
        assert parsed.couplings[0].i == 0 and parsed.couplings[0].j == 1
        assert parsed.couplings[0].eta_loss == pytest.approx(0.98, abs=1e-12)

    def test_missing_n_modes_raises(self):
        with pytest.raises(ValueError, match="n_modes"):
            load_topology_config({"squeezing_r": [0.2, 0.3]})

    def test_invalid_loss_range_raises(self):
        cfg = {
            "n_modes": 2,
            "squeezing_r": [0.2, 0.3],
            "loss": [1.0, 1.2],
        }
        with pytest.raises(ValueError, match="loss entries must be within"):
            load_topology_config(cfg)

    def test_invalid_coupling_indices_raise(self):
        cfg = {
            "n_modes": 3,
            "squeezing_r": [0.2, 0.3, 0.4],
            "loss": [1.0, 1.0, 1.0],
            "couplings": [{"i": 0, "j": 4, "theta": 0.2}],
        }
        with pytest.raises(ValueError, match="indices out of range"):
            load_topology_config(cfg)


class TestTopologySimulation:
    def test_zero_coupling_reduces_to_independent_multimode(self):
        r = np.array([0.4, 0.7, 0.2], dtype=float)
        squeeze_phi = np.array([0.0, 0.1, -0.2], dtype=float)
        theta = np.array([0.2, -0.3, 0.1], dtype=float)
        loss = np.array([1.0, 0.85, 0.65], dtype=float)

        top_cfg = {
            "n_modes": 3,
            "squeezing_r": r.tolist(),
            "squeezing_phi": squeeze_phi.tolist(),
            "phase_shifts": theta.tolist(),
            "loss": loss.tolist(),
            "couplings": [
                {"i": 0, "j": 1, "theta": 0.0, "phi": 0.0, "eta_loss": 1.0},
                {"i": 1, "j": 2, "theta": 0.0},
            ],
        }

        top = simulate_topology(top_cfg)
        base = run_multimode(
            r=r,
            theta=theta,
            eta_loss=loss,
            squeeze_theta=squeeze_phi,
            n_modes=3,
        )

        assert np.allclose(top.var_x, base.var_x, atol=1e-9, rtol=1e-8)
        assert np.allclose(top.var_p, base.var_p, atol=1e-9, rtol=1e-8)
        assert np.allclose(top.mean_photon, base.mean_photon, atol=1e-9, rtol=1e-8)
        assert np.allclose(top.observed_sq_db, base.observed_sq_db, atol=1e-9, rtol=1e-8)
        assert np.allclose(
            top.observed_antisq_db,
            base.observed_antisq_db,
            atol=1e-9,
            rtol=1e-8,
        )
        assert np.allclose(top.neighbor_cov_x, 0.0, atol=1e-8)
        assert np.allclose(top.neighbor_cov_p, 0.0, atol=1e-8)

    def test_two_mode_snapshot_with_coupling(self):
        cfg = {
            "n_modes": 2,
            "squeezing_r": [0.5, 0.2],
            "phase_shifts": [0.0, 0.0],
            "loss": [1.0, 1.0],
            "couplings": [{"i": 0, "j": 1, "theta": np.pi / 4, "phi": 0.0}],
        }
        result = simulate_topology(cfg)

        expected_cov = np.array(
            [
                [0.259549871802, -0.075610151216, 0.0, 0.0],
                [-0.075610151216, 0.259549871802, 0.0, 0.0],
                [0.0, 0.0, 1.052526631525, 0.306614282704],
                [0.0, 0.0, 0.306614282704, 1.052526631525],
            ],
            dtype=float,
        )

        assert np.allclose(result.covariance, expected_cov, atol=1e-9, rtol=1e-8)
        assert result.neighbor_cov_x[0] == pytest.approx(-0.075610151216, abs=1e-9)
        assert result.neighbor_cov_p[0] == pytest.approx(0.306614282704, abs=1e-9)
