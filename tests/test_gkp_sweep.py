"""Tests for the GKP toy sweep script."""

from pathlib import Path

import numpy as np

from scripts.run_gkp_sweep import logical_error_proxy, run_gkp_sweep


def test_logical_error_monotonic_with_noise_and_squeezing() -> None:
    noise = np.linspace(0.02, 0.7, 40)
    r = np.array([0.2, 0.6, 1.0])[:, None]
    proxy = logical_error_proxy(r, noise[None, :])

    # Higher noise -> higher logical error
    assert np.all(np.diff(proxy[0]) > -1e-12)
    # Higher squeezing -> lower logical error
    assert np.all(proxy[0] > proxy[2])


def test_run_gkp_sweep_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "gkp_proxy.png"
    produced = run_gkp_sweep(output_path=out, squeezing_points=6, noise_points=32)
    assert produced == out
    assert out.exists()
    assert out.stat().st_size > 0
