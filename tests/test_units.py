"""Tests for shared unit conversion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_optical_bus.units import (
    db_to_eta,
    eta_to_db,
    observed_squeezing_from_cov,
    sf_cov_to_vacuum05,
)


def test_db_eta_roundtrip_scalar_and_vector():
    assert db_to_eta(0.0) == pytest.approx(1.0, abs=1e-12)
    assert eta_to_db(1.0) == pytest.approx(0.0, abs=1e-12)

    losses = np.array([0.0, 1.0, 3.0, 10.0], dtype=float)
    etas = db_to_eta(losses)
    assert np.allclose(eta_to_db(etas), losses, rtol=1e-12, atol=1e-12)


def test_sf_cov_conversion_and_observed_squeezing_vacuum():
    # SF gaussian backend vacuum covariance uses hbar=2 -> identity(2)
    sf_cov = np.eye(2, dtype=float)
    cov_05 = sf_cov_to_vacuum05(sf_cov)
    assert np.allclose(cov_05, 0.5 * np.eye(2), atol=1e-12)

    sq_db, anti_sq_db = observed_squeezing_from_cov(cov_05)
    assert sq_db == pytest.approx(0.0, abs=1e-12)
    assert anti_sq_db == pytest.approx(0.0, abs=1e-12)


def test_observed_squeezing_from_diagonal_cov():
    # Simple squeezed / anti-squeezed diagonal covariance in vacuum=0.5 convention.
    cov = np.diag([0.25, 1.0]).astype(float)
    sq_db, anti_sq_db = observed_squeezing_from_cov(cov)
    assert sq_db == pytest.approx(3.0102999566, rel=1e-9)
    assert anti_sq_db == pytest.approx(3.0102999566, rel=1e-9)
