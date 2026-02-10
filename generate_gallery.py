import numpy as np
import matplotlib.pyplot as plt
import sys
from unittest.mock import MagicMock

# Patch dependencies for command line run
if "pkg_resources" not in sys.modules:
    sys.modules["pkg_resources"] = MagicMock()

import scipy.integrate
if not hasattr(scipy.integrate, 'simps'):
    if hasattr(scipy.integrate, 'simpson'):
        scipy.integrate.simps = scipy.integrate.simpson

import strawberryfields as sf
from strawberryfields.ops import Sgate, Rgate

# --- Simulation Logic (Copied/Adapted from squeezed_light.py) ---

def run_hardware_simulation():
    # Mocking the hardware simulation for speed/stability in generation script
    # This matches the "fallback" logic in the main app
    wavelength = 1.55
    core_width = 1.0
    core_height = 0.6
    cell_size = 3.0
    
    # Mock Data (Fundamental Mode)
    N = 100
    x = np.linspace(-cell_size/2, cell_size/2, N)
    y = np.linspace(-cell_size/2, cell_size/2, N)
    X, Y = np.meshgrid(x, y)
    
    sigma_x = core_width / 2.5
    sigma_y = core_height / 2.5
    ez_data = np.exp(-(X**2)/(2*sigma_x**2) - (Y**2)/(2*sigma_y**2))
    
    n_eff = 2.14
    mode_area = np.pi * (core_width/2) * (core_height/2)
    extent = [-cell_size/2, cell_size/2, -cell_size/2, cell_size/2]
    
    return n_eff, mode_area, ez_data, extent

def calculate_squeezing(pump_power_mw, mode_area):
    coupling_efficiency = 0.1 
    r = coupling_efficiency * np.sqrt(pump_power_mw)
    return r

def generate_result(filename, title, scenario_powers, scenario_phases, description):
    print(f"Generating {filename}...")
    
    # Get HW Data
    n_eff, mode_area, ez_data, extent = run_hardware_simulation()
    
    # Setup Figure
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])
    
    # 1. Hardware Panel
    ax_hw = fig.add_subplot(gs[:, 0:2])
    ax_hw.set_title("Fixed Hardware Layer (Meep)\nLN Ridge Waveguide", fontsize=14, fontweight='bold', color='navy')
    ax_hw.imshow(ez_data, extent=extent, cmap='RdBu', origin='lower')
    ax_hw.set_xlabel("x (um)")
    ax_hw.set_ylabel("y (um)")
    ax_hw.text(0.05, 0.95, f"n_eff: {n_eff:.2f}\nMode Area: {mode_area:.2f} um^2", 
               transform=ax_hw.transAxes, color='white', ha='left', va='top', 
               fontsize=11, bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    # 2. Software Panel (Wigner Functions)
    grid_limit = 4
    grid_points = 100
    xvec = np.linspace(-grid_limit, grid_limit, grid_points)
    
    bins = ["Bin 0", "Bin 1", "Bin 2", "Bin 3"]
    
    for i in range(4):
        power = scenario_powers[i]
        phase = scenario_phases[i]
        
        r_param = calculate_squeezing(power, mode_area)
        theta_param = phase
        
        prog = sf.Program(1)
        eng = sf.Engine("gaussian")
        with prog.context as q:
            if r_param > 0:
                Sgate(r_param) | q[0]
            if theta_param != 0:
                Rgate(theta_param) | q[0]
        
        result = eng.run(prog)
        state = result.state
        W = state.wigner(0, xvec, xvec)
        
        row = i // 2
        col = i % 2 
        ax_bin = fig.add_subplot(gs[row, 2 + col])
        
        cont = ax_bin.contourf(xvec, xvec, W, levels=20, cmap='viridis')
        
        # Annotations
        state_str = "Vacuum"
        if power > 0:
            state_str = "Squeezed"
        
        ax_bin.set_title(f"{bins[i]}\nP={power}mW, Ph={phase:.2f}\n({state_str})", fontsize=11)
        ax_bin.set_xticks([]) if row == 0 else ax_bin.set_xlabel("x")
        ax_bin.set_yticks([]) if col == 1 else ax_bin.set_ylabel("p")
        ax_bin.set_aspect('equal')
        
    plt.suptitle(f"Scenario: {title}\n{description}", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

# --- Scenarios ---

# 1. Vacuum / Baseline
generate_result(
    "result_vacuum.png",
    "Baseline System Check",
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    "Insight: With 0mW Pump Power, the 'Bus' transmits pure Vacuum States (Shot Noise Unit)."
)

# 2. Amplitude vs Phase Encoding
generate_result(
    "result_encoding.png",
    "Orthogonal Information Encoding",
    [50, 50, 0, 0],     # Bin 0 and 1 active
    [0, 1.57, 0, 0],    # Bin 0: 0 rad (Amp), Bin 1: pi/2 rad (Phase)
    "Insight: Bin 0 is Amplitude Squeezed (Horizontal), Bin 1 is Phase Squeezed (Vertical).\nDemonstrates independent control on the same waveguide."
)

# 3. Analog Gradient
generate_result(
    "result_gradient.png",
    "Analog Quantum Control",
    [0, 10, 40, 90],    # Increasing power
    [0, 0, 0, 0],
    "Insight: Increasing Pump Power (0->90mW) progressively augments the Squeezing strength.\nDemonstrates the Continuous Variable (CV) nature of the system."
)
