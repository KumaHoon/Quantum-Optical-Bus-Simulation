# Architecture Guide

This document explains the architecture of the current codebase in terms of module
responsibilities and execution flow.
It is positioned as a loop-based optical workflow scaffold for cross-platform transfer, where data assimilation and estimation feed FPGA-friendly control updates in a closed-loop deployment path.

## Review-boundary note

For README-focused review, core evidence is the control trio (`calibration_demo.gif`, `dashboard_decoherence.png`, one `sweep_*` output).
Multi-mode, topology, digital-twin dashboard, and drift-recovery figures are roadmap-strengthening extensions.

## Orchestrator role

`src/quantum_optical_bus/calibration_app.py` is the application orchestrator.
It:

- Builds the Streamlit layout and sectioned workflow.
- Collects all user inputs (pump power, phase, loss, topology settings, etc.).
- Calls computational modules behind cached helpers and renders diagnostics:
  - single-mode Wigner + squeezing metrics,
  - variance / photon-distribution plots,
  - independent multi-mode sweeps,
  - topology runs,
  - digital twin fit and control sweep results.

It does not contain the physics model itself; it orchestrates module execution and visualization.

## Module-by-module map

### 1) `src/quantum_optical_bus/quantum.py`

Primary function:

- `run_single_mode(r, theta, eta_loss, xvec)`

Behavior:

- Builds a one-mode Strawberry Fields Gaussian circuit with optional:
  - `Sgate(r)`
  - `Rgate(theta)`
  - `LossChannel(eta_loss)`
- Runs via `sf.Engine("gaussian")`.
- Returns `QuantumResult` containing:
  - `W`: Wigner snapshot on supplied `xvec`,
  - `mean_photon`, `var_x`, `var_p`,
  - observed squeezing and anti-squeezing in dB from output covariance.

Use in workflow:

- Primary single-mode path for vacuum/calibration/decoherence tabs in the dashboard.

### 2) `src/quantum_optical_bus/multimode.py`

Primary function:

- `run_multimode(r, theta, eta_loss, n_modes=None, squeeze_theta=None, xvec=None, wigner_mode=None)`

Behavior:

- Vectorizes scalar/sequence inputs into per-mode arrays.
- Builds an independent Gaussian circuit of `n` modes:
  - per-mode `Sgate`,
  - per-mode `Rgate`,
  - per-mode `LossChannel`.
- No inter-mode couplings are added in this module.
- Returns per-mode `mean_photon`, `var_x`, `var_p`, observed squeezing arrays.
- Optionally computes one selected mode Wigner if `wigner_mode` is provided.

Use in workflow:

- Phase 4 "Multi-mode" tab shows per-bin squeezing/variance trajectories.

### 3) `src/quantum_optical_bus/tdm_topology.py`

Primary function:

- `simulate_topology(config)`

Behavior:

- Loads configuration from typed config, dict, JSON, or YAML.
- Normalizes per-mode arrays and coupling list.
- For each mode applies local Gaussian operations (`Sgate`, `Rgate`, `LossChannel`).
- Applies ordered inter-bin couplings from config edges via `BSgate`.
- Returns `TopologySimulationResult` including:
  - covariance,
  - per-mode variances and squeezing,
  - neighbor covariances and normalized correlation matrices.

Use in workflow:

- Phase 4 topology path and static chain/coupling visualization.

### 4) `src/quantum_optical_bus/estimation.py`

Primary function:

- `fit_eta_and_loss(data, model="auto")`

Behavior:

- Accepts in-memory dict-like data or CSV path.
- Requires `pump_power_mw` plus either variance or squeezing observations.
- Predicts variances from parameters `(eta, loss_db)`:
  - `r = eta * sqrt(pump_power_mw)`
  - transmissivity from loss in dB,
  - analytic var model in SNU (vacuum=0.5),
  - then least-squares fit via SciPy.
- Returns `(eta_hat, loss_hat, diagnostics)`.

Use in workflow:

- Digital twin estimation tab and fit summary metrics.

### 5) `src/quantum_optical_bus/control.py`

Primary routines:

- `simulate_phase_drift`
- `apply_feedback_with_latency`

Behavior:

- Generates a wrapped random-walk phase trajectory (with optional drift).
- Applies an estimator/controller abstraction with configurable latency.
- Produces residual error arrays and metrics such as RMS residual phase error and retention proxy.

Use in workflow:

- Latency and control sensitivity sweep visualizations in the digital twin tab.

### 6) `src/quantum_optical_bus/hardware.py`

Primary function:

- `run_hardware_simulation(config=None)`

Behavior:

- Attempts a Meep eigenmode solve when `meep` is available.
- On failure or when Meep is unavailable, silently falls back to `_mock_mode`.
- Returns:
  - effective index,
  - mode-area estimate,
  - mode profile array,
  - extent for plotting.

Use in workflow:

- Phase 1 "The Device" visualization in the dashboard.
- **Current scope note**: this path is display/inference-oriented and does not yet close the mapping to `r` calibration in the live UI.

### 7) FPGA contract boundary (deployment edge)

The project keeps FPGA deployment risk narrow by defining a fixed contract:

- Host side:
  - derive model and control updates from `estimation.py` outputs
  - quantize control values before dispatch
- RTL side:
  - `hdl/feedforward_lut.sv` expects signed Q1.15 input and returns signed Q1.15 output
  - fixed latency is `PIPE_LATENCY = 2` cycles in the reference block
- Evidence:
  - `hdl/vectors/stimulus.mem`
  - `hdl/vectors/expected.mem`
  - `hdl/vectors/contract.json`
  - `make -C hdl sim` (VCD + latency print)

This lets the loop-based photonic stack evolve independently in the optics/sensor path while keeping the FPGA boundary explicitly testable.

## Data/flow summary (end-to-end)

```mermaid
flowchart TD
  UserInput[User inputs in Streamlit] --> UI["calibration_app.py"]
  UI -->|P -> r| Map["interface.py"]
  Map -->|eta*sqrt(P)| Single["quantum.py run_single_mode"]
  Map --> Multi["multimode.py run_multimode"]
  Map --> Topo["tdm_topology.py simulate_topology"]

  UI --> Units["units.py db_to_eta / sf_cov_to_vacuum05"]
  UI --> HWRaw["hardware.py run_hardware_simulation"]
  HWRaw --> UI

  Units --> Single
  Units --> Multi
  Units --> Topo

  Single --> UI
  Multi --> UI
  Topo --> UI

  UI --> Data["estimation.py fit_eta_and_loss"]
  UI --> Ctrl["control.py latency + drift"]
  UI --> HdlContract["hdl/contract.json and vectors/*.mem (Q1.15)"]
  HdlContract --> Feedforward[feedforward_lut.sv]
  Data --> UI
  Ctrl --> UI
``` 

## Placeholder and roadmap notes

- Calibration coupling coefficient $\eta$ in `interface.py` is source-level and tuned for expected operating conditions in this MVP.
- Hardware-to-mapping closure from measured mode properties to squeezing gain is not yet implemented.
- `hardware.py` currently includes an exploratory path for future calibrated flows and a stable fallback path.
- Topology and control modules are MVP simulations with future extensions for full timing/jitter and actuator hardware integration.
- `calibration_app.py` includes display-focused and educational annotations intended as a white-box workflow.
