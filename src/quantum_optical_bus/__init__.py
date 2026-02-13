"""
Quantum Optical Bus Simulation

A hybrid quantum-classical simulation demonstrating
"One Waveguide (Hardware), Infinite States (Software)".
"""

from .hardware import run_hardware_simulation, WaveguideConfig
from .interface import calculate_squeezing

__all__ = [
    "run_hardware_simulation",
    "WaveguideConfig",
    "calculate_squeezing",
]

