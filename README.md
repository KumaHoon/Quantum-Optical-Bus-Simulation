# Modern Physics Simulation with Marimo and UV

This project implements a reactive physics simulation environment using [Marimo](https://marimo.io) and manages dependencies with [uv](https://github.com/astral-sh/uv).

## Features

- **Reactive UI**: Adjust parameters like waveguide width and refractive index using sliders.
- **Real-time Visualization**: Instant feedback on mode profiles.
- **SOLID Architecture**: Clean separation of physics logic, visualization, and UI.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

## Quick Start

1. **Clone/Open the project** in your terminal.
2. **Run the simulation**:
   ```bash
   uv run marimo edit simulation.py
   ```
   This command will:
   - Create a virtual environment.
   - Install all necessary dependencies (`numpy`, `scipy`, `matplotlib`, `marimo`).
   - Launch the Marimo editor in your browser.

## Project Structure

- `simulation.py`: The main application file (Marimo notebook).
- `pyproject.toml`: Dependency configuration.
