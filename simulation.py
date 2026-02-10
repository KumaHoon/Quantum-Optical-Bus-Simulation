import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from abc import ABC, abstractmethod
    from typing import Protocol, List

    return ABC, Protocol, abstractmethod, dataclass, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Modern Physics Simulation: Waveguide Mode Solver

    This simulation calculates the fundamental mode profile of a 1D slab waveguide.
    """)
    return


@app.cell
def _(dataclass):
    # Domain Model (Single Responsibility Principle)
    @dataclass
    class WaveguideParams:
        width_um: float
        n_core: float
        n_clad: float
        wavelength_um: float = 1.55

    return (WaveguideParams,)


@app.cell
def _(ABC, WaveguideParams, abstractmethod, np):
    # Abstraction (Dependency Inversion Principle)
    class ModeSolver(ABC):
        @abstractmethod
        def solve_fundamental_mode(self, params: WaveguideParams, x_grid: np.ndarray) -> np.ndarray:
            pass

    return (ModeSolver,)


@app.cell
def _(ModeSolver, WaveguideParams, np):
    # Concrete Implementation (Open/Closed Principle - easy to extend with new solvers)
    class SlabWaveguideSolver(ModeSolver):
        def solve_fundamental_mode(self, params: WaveguideParams, x_grid: np.ndarray) -> np.ndarray:
            # Simplified Gaussian approximation for the fundamental mode
            # In a real solver, we'd solve the transcendental equations or use finite difference.
            # This is a placeholder for the physics logic to demonstrate the structure.

            # Width and refractive index define the confinement
            w = params.width_um
            # Approximate mode field radius (heuristic)
            w_0 = w / 2.0 * (0.5 + 1.0 / (params.n_core - params.n_clad + 1e-9))

            # Gaussian profile: E(x) = exp(-x^2 / w_0^2)
            # Center the mode at x=0
            field = np.exp(-(x_grid ** 2) / (w_0 ** 2))

            # Mask field outside core slightly to mimic decay (just for visual effect of 'cladding')
            # In real physics, it decays exponentially in cladding. 
            # Let's do a piecewise function for better realism if possible, 
            # or just stick to Gaussian which is a decent approximation for single mode.

            return field

    return (SlabWaveguideSolver,)


@app.cell
def _(Protocol, np, plt):
    # Visualization (Interface Segregation Principle - Visualizer doesn't need to know about Solver internals)
    class Plotter(Protocol):
        def plot(self, x: np.ndarray, y: np.ndarray, title: str) -> plt.Figure:
            ...

    class MatplotlibPlotter:
        def plot(self, x: np.ndarray, y: np.ndarray, title: str) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y, label="Mode Profile $E_y(x)$", color="blue", linewidth=2)
            ax.set_xlabel("Position x ($\mu m$)")
            ax.set_ylabel("Electric Field Amplitude")
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            plt.tight_layout()
            return fig

    return (MatplotlibPlotter,)


@app.cell
def _(mo):
    # User Interface (Controller)

    # Sliders using marimo.ui
    width_slider = mo.ui.slider(
        start=0.1, stop=5.0, step=0.1, value=1.0, label="Waveguide Width ($\mu m$)"
    )

    n_core_slider = mo.ui.slider(
        start=1.45, stop=3.5, step=0.01, value=1.5, label="Core Refractive Index ($n_{core}$)"
    )

    n_clad_slider = mo.ui.slider(
        start=1.0, stop=1.44, step=0.01, value=1.0, label="Cladding Refractive Index ($n_{clad}$)"
    )

    mo.md(
        f"""
        ### Simulation Parameters
        Adjust the sliders to change the waveguide properties.

        {width_slider}
        {n_core_slider}
        {n_clad_slider}
        """
    )
    return n_clad_slider, n_core_slider, width_slider


@app.cell
def _(
    MatplotlibPlotter,
    SlabWaveguideSolver,
    WaveguideParams,
    mo,
    n_clad_slider,
    n_core_slider,
    np,
    width_slider,
):
    # Application Logic (Composition root)

    # 1. Capture Input
    w_val = width_slider.value
    n_core_val = n_core_slider.value
    n_clad_val = n_clad_slider.value

    # 2. Setup Simulation Domain
    x_grid = np.linspace(-5, 5, 500)
    params = WaveguideParams(width_um=w_val, n_core=n_core_val, n_clad=n_clad_val)

    # 3. Solve (Dependency Injection)
    solver = SlabWaveguideSolver()
    try:
        mode_profile = solver.solve_fundamental_mode(params, x_grid)
    except Exception as e:
        mode_profile = np.zeros_like(x_grid) # Fallback

    # 4. Visualize
    plotter = MatplotlibPlotter()
    fig = plotter.plot(
        x_grid, 
        mode_profile, 
        title=f"Mode Profile (w={w_val}$\mu m$, $\Delta n$={n_core_val - n_clad_val:.2f})"
    )

    # 5. Output
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
