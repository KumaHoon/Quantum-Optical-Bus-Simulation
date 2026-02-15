"""
Topology-driven multi-mode Gaussian simulation for TDM loop experiments.

This module adds inter-bin coupling via beam splitters and exposes a
configuration-driven API:
    simulate_topology(config) -> covariance + correlation summary
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
import json

import numpy as np

# Compat patches must be applied before importing Strawberry Fields.
import quantum_optical_bus.compat  # noqa: F401

import strawberryfields as sf
from strawberryfields.ops import BSgate, LossChannel, Rgate, Sgate

from quantum_optical_bus.units import observed_squeezing_from_cov, sf_cov_to_vacuum05


@dataclass(frozen=True)
class BeamSplitterCoupling:
    """Beam-splitter edge between two bins (modes)."""

    i: int
    j: int
    theta: float
    phi: float = 0.0
    eta_loss: float = 1.0


@dataclass(frozen=True)
class TDMTopologyConfig:
    """Config schema for topology simulation."""

    n_modes: int
    squeezing_r: np.ndarray
    squeezing_phi: np.ndarray
    phase_shifts: np.ndarray
    loss: np.ndarray
    couplings: tuple[BeamSplitterCoupling, ...]


class TopologySimulationResult(NamedTuple):
    """Simulation outputs and compact correlation summary."""

    covariance: np.ndarray
    mean_photon: np.ndarray
    var_x: np.ndarray
    var_p: np.ndarray
    observed_sq_db: np.ndarray
    observed_antisq_db: np.ndarray
    neighbor_cov_x: np.ndarray
    neighbor_cov_p: np.ndarray
    corr_x: np.ndarray
    corr_p: np.ndarray
    coupling_proxy: np.ndarray


def _to_vector(values: Any, n_modes: int, default: float, name: str) -> np.ndarray:
    if values is None:
        return np.full(n_modes, default, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(n_modes, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D array-like")
    if arr.size == 1:
        return np.full(n_modes, float(arr[0]), dtype=float)
    if arr.size != n_modes:
        raise ValueError(f"{name} has length {arr.size}, expected {n_modes}")
    return arr.astype(float, copy=False)


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only when yaml missing
            raise ImportError("YAML config requires PyYAML to be installed") from exc
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    raise ValueError(f"Unsupported config file extension: {suffix}")


def load_topology_config(
    config: TDMTopologyConfig | dict[str, Any] | str | Path,
) -> TDMTopologyConfig:
    """Load and validate topology config from dict/path/dataclass."""
    if isinstance(config, TDMTopologyConfig):
        return config

    raw: dict[str, Any]
    if isinstance(config, (str, Path)):
        raw = _read_config_file(Path(config))
    elif isinstance(config, dict):
        raw = config
    else:
        raise TypeError("config must be TDMTopologyConfig, dict, str, or Path")

    if "n_modes" not in raw:
        raise ValueError("config must define n_modes")
    n_modes = int(raw["n_modes"])
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")

    squeezing_r = _to_vector(raw.get("squeezing_r", raw.get("r")), n_modes, 0.0, "squeezing_r")
    squeezing_phi = _to_vector(
        raw.get("squeezing_phi", raw.get("squeeze_theta")),
        n_modes,
        0.0,
        "squeezing_phi",
    )
    phase_shifts = _to_vector(
        raw.get("phase_shifts", raw.get("theta")), n_modes, 0.0, "phase_shifts"
    )
    loss = _to_vector(raw.get("loss", raw.get("eta_loss")), n_modes, 1.0, "loss")

    if np.any((loss < 0.0) | (loss > 1.0)):
        raise ValueError("loss entries must be within [0, 1]")

    couplings_raw = raw.get("couplings", [])
    couplings: list[BeamSplitterCoupling] = []
    for idx, edge in enumerate(couplings_raw):
        if not isinstance(edge, dict):
            raise ValueError(f"couplings[{idx}] must be an object")
        i = int(edge["i"])
        j = int(edge["j"])
        if i == j:
            raise ValueError(f"couplings[{idx}] has identical mode indices")
        if i < 0 or i >= n_modes or j < 0 or j >= n_modes:
            raise ValueError(f"couplings[{idx}] indices out of range for n_modes={n_modes}")
        eta_loss = float(edge.get("eta_loss", edge.get("loss", 1.0)))
        if eta_loss < 0.0 or eta_loss > 1.0:
            raise ValueError(f"couplings[{idx}].eta_loss must be within [0, 1]")
        couplings.append(
            BeamSplitterCoupling(
                i=i,
                j=j,
                theta=float(edge.get("theta", 0.0)),
                phi=float(edge.get("phi", 0.0)),
                eta_loss=eta_loss,
            )
        )

    return TDMTopologyConfig(
        n_modes=n_modes,
        squeezing_r=squeezing_r,
        squeezing_phi=squeezing_phi,
        phase_shifts=phase_shifts,
        loss=loss,
        couplings=tuple(couplings),
    )


def simulate_topology(
    config: TDMTopologyConfig | dict[str, Any] | str | Path,
) -> TopologySimulationResult:
    """Simulate a topology with optional inter-bin beam-splitter couplings."""
    cfg = load_topology_config(config)
    n = cfg.n_modes

    prog = sf.Program(n)
    with prog.context as q:
        # Local per-bin operations.
        for mode in range(n):
            if cfg.squeezing_r[mode] != 0.0:
                Sgate(cfg.squeezing_r[mode], cfg.squeezing_phi[mode]) | q[mode]
            if cfg.phase_shifts[mode] != 0.0:
                Rgate(cfg.phase_shifts[mode]) | q[mode]
            if cfg.loss[mode] < 1.0:
                LossChannel(cfg.loss[mode]) | q[mode]

        # Inter-bin couplings (MVP).
        for edge in cfg.couplings:
            if edge.theta != 0.0 or edge.phi != 0.0:
                BSgate(edge.theta, edge.phi) | (q[edge.i], q[edge.j])
            if edge.eta_loss < 1.0:
                LossChannel(edge.eta_loss) | q[edge.i]
                LossChannel(edge.eta_loss) | q[edge.j]

    state = sf.Engine("gaussian").run(prog).state
    cov = sf_cov_to_vacuum05(state.cov())

    mean_photon = np.zeros(n, dtype=float)
    var_x = np.zeros(n, dtype=float)
    var_p = np.zeros(n, dtype=float)
    observed_sq_db = np.zeros(n, dtype=float)
    observed_antisq_db = np.zeros(n, dtype=float)

    for mode in range(n):
        mean_photon[mode] = float(state.mean_photon(mode)[0])
        x_idx = mode
        p_idx = n + mode
        block = cov[np.ix_([x_idx, p_idx], [x_idx, p_idx])]
        var_x[mode] = float(cov[x_idx, x_idx])
        var_p[mode] = float(cov[p_idx, p_idx])
        observed_sq_db[mode], observed_antisq_db[mode] = observed_squeezing_from_cov(block)

    x_cov = cov[:n, :n]
    p_cov = cov[n:, n:]
    corr_x = _cov_to_corr(x_cov)
    corr_p = _cov_to_corr(p_cov)

    neighbor_cov_x = np.array([cov[i, i + 1] for i in range(n - 1)], dtype=float)
    neighbor_cov_p = np.array([cov[n + i, n + i + 1] for i in range(n - 1)], dtype=float)
    coupling_proxy = np.abs(corr_x - np.eye(n)) + np.abs(corr_p - np.eye(n))

    return TopologySimulationResult(
        covariance=cov,
        mean_photon=mean_photon,
        var_x=var_x,
        var_p=var_p,
        observed_sq_db=observed_sq_db,
        observed_antisq_db=observed_antisq_db,
        neighbor_cov_x=neighbor_cov_x,
        neighbor_cov_p=neighbor_cov_p,
        corr_x=corr_x,
        corr_p=corr_p,
        coupling_proxy=coupling_proxy,
    )
