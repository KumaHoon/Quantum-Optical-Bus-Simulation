# Evidence Pack

## Control Constraints Sweeps Deliverable

One-sentence claim: The repository now produces reproducible control-constraint sweeps that quantify how control latency and fixed-point quantization affect residual phase performance.

## HDL Nonlinear Feedforward Deliverable

One-sentence claim: The repository now includes a fixed-point LUT-based feedforward HDL block with reproducible simulation and golden-vector verification.

## GKP Toy Deliverable

One-sentence claim: The repository now includes a minimal, explainable GKP toy model linking squeezing/noise proxy to logical error proxy.

## Stability / Automation Deliverable

One-sentence claim: The repository now simulates 24h drift with estimation and closed-loop gain updates, demonstrating recovery behavior.

## Artifacts
- `assets/sweep_latency.png` - latency sweep image (bins vs RMS residual and retention proxy)
- `assets/sweep_quantization.png` - quantization sweep image (bit width vs RMS residual)
- `scripts/generate_control_sweeps.py` - single reproducible command to regenerate both sweep artifacts
- `assets/gkp_proxy.png` - GKP proxy image (squeezing + noise sweep)
- `assets/drift_recovery.png` - 24h drift/estimation/controller recovery demonstration
- `scripts/run_gkp_sweep.py` - reproducible GKP toy proxy script
- `notebooks/04_gkp_ec_toy.ipynb` - notebook equivalent of GKP toy workflow
- `hdl/feedforward_lut.sv` - fixed-point feedforward LUT pipeline
- `hdl/tb_feedforward_lut.sv` - checks outputs and writes `waves.vcd`
- `hdl/Makefile` - `make -C hdl sim` target for compile/run
- `hdl/vectors/stimulus.mem` - generated stimulus vectors
- `hdl/vectors/expected.mem` - generated expected output vectors
- `scripts/export_golden_vectors.py` - reproducible golden-vector generator for the TB
- `tests/test_hdl_feedforward.py` - regression check for vector generation artifacts
- `tests/test_gkp_sweep.py` - regression checks for GKP proxy monotonicity and artifact generation
- `scripts/simulate_24h_drift.py` - reproducible drift+estimator+controller simulation
- `tests/test_drift_automation.py` - deterministic drift profile and artifact check

## Reproduce
- `make test` (or `python -m pytest -q`)
- `make lint` (or `python -m ruff check scripts/export_golden_vectors.py tests/test_control_sweeps.py tests/test_hdl_feedforward.py tests/test_gkp_sweep.py tests/test_drift_automation.py`)
- `python scripts/generate_control_sweeps.py`
- `python scripts/export_golden_vectors.py --output-dir hdl/vectors --count 128 --seed 13`
- `make -C hdl sim` (requires Icarus Verilog: `iverilog`, `vvp`)
- `python scripts/run_gkp_sweep.py`
- `python scripts/simulate_24h_drift.py`

## Claim Mapping
- Control constraints: `assets/sweep_latency.png`, `assets/sweep_quantization.png`, and `scripts/generate_control_sweeps.py`.
- Nonlinear feedforward / FPGA: `hdl/feedforward_lut.sv`, `hdl/tb_feedforward_lut.sv`, `hdl/Makefile`, and `waves.vcd`.
- World-modeling/control co-design: existing control sweeps use drift trajectories from `quantum_optical_bus.control`.
- GKP / fault tolerance: `assets/gkp_proxy.png`, `scripts/run_gkp_sweep.py`, `notebooks/04_gkp_ec_toy.ipynb`.
- Drift / automation: `assets/drift_recovery.png`, `scripts/simulate_24h_drift.py`.
- Stability/automation: deterministic seeds and reproducible commands (`python -m pytest -q`, `python scripts/generate_control_sweeps.py`, `python scripts/export_golden_vectors.py`, `python scripts/run_gkp_sweep.py`, `python scripts/simulate_24h_drift.py`).

