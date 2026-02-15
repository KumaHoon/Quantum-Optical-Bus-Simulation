"""
Quantum Optical Bus Simulation

A hybrid quantum-classical simulation demonstrating
"One Waveguide (Hardware), Infinite States (Software)".
"""

from .hardware import run_hardware_simulation, WaveguideConfig
from .interface import calculate_squeezing
from .multimode import run_multimode, MultiModeResult
from .quantum import run_single_mode, QuantumResult
from .estimation import fit_eta_and_loss
from .control import simulate_phase_drift, apply_feedback_with_latency
from .tdm_topology import (
    BeamSplitterCoupling,
    TDMTopologyConfig,
    TopologySimulationResult,
    load_topology_config,
    simulate_topology,
)

__all__ = [
    "run_hardware_simulation",
    "WaveguideConfig",
    "calculate_squeezing",
    "run_single_mode",
    "QuantumResult",
    "run_multimode",
    "MultiModeResult",
    "fit_eta_and_loss",
    "simulate_phase_drift",
    "apply_feedback_with_latency",
    "BeamSplitterCoupling",
    "TDMTopologyConfig",
    "TopologySimulationResult",
    "load_topology_config",
    "simulate_topology",
]

