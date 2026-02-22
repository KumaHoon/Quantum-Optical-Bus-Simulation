# Evidence Pack

## MVP evidence boundary (README-facing)

For reviewer acceptance, the repository treats the following artifacts as mandatory core evidence:

- `assets/web/calibration_demo.gif`
- `assets/web/dashboard_decoherence.png`
- `assets/web/sweep_latency.png` or `assets/web/sweep_quantization.png` (either is acceptable as a control-constraint sensitivity example)

Everything else is non-blocking: optional appendix, roadmap, or polishing material.

## Core evidence mapping for README route

- Core claim 1: intrinsic-vs-observed calibration behavior is visible in `assets/web/calibration_demo.gif`.
- Core claim 2: loss realism is visible in `assets/web/dashboard_decoherence.png`.
- Core claim 3: control constraint sensitivity is visible in `assets/web/sweep_latency.png` (or `assets/web/sweep_quantization.png`).

## Control Constraints Sweeps Deliverable

One-sentence claim: The repository produces reproducible control-constraint sweeps that quantify how control latency and fixed-point quantization affect residual phase performance under loop-like operation.

## HDL Nonlinear Feedforward Deliverable

One-sentence claim: The repository includes a fixed-point LUT-based feedforward HDL block with reproducible simulation and golden-vector verification.

## Stability / Automation Deliverable

One-sentence claim: The repository simulates 24h drift with estimation and closed-loop gain updates, demonstrating recovery behavior.

## Artifacts

### Core artifacts (mandatory for MVP review)

- `assets/web/sweep_latency.png` (review) and `assets/paper/sweep_latency.png` (publication) - latency sensitivity map
- `assets/web/sweep_quantization.png` (review) and `assets/paper/sweep_quantization.png` (publication) - quantization sensitivity map
- `assets/web/calibration_demo.gif` - calibration workflow and monotonic loss check
- `assets/web/dashboard_decoherence.png` - pure vs lossy comparison
- `hdl/feedforward_lut.sv` - fixed-point feedforward LUT pipeline
- `hdl/tb_feedforward_lut.sv` - HDL verification testbench
- `hdl/Makefile` - `make -C hdl sim`
- `hdl/vectors/stimulus.mem` - generated stimuli
- `hdl/vectors/expected.mem` - expected outputs
- `scripts/export_golden_vectors.py` - reproducible golden-vector generator
- `tests/test_hdl_feedforward.py` - artifact-creation regression check
- `scripts/simulate_24h_drift.py` - deterministic drift+estimator+controller simulation
- `tests/test_drift_automation.py` - deterministic drift/automation artifact test
- `hdl/vectors/contract.json` - fixed-point/latency/LUT manifest metadata

### Optional / roadmap artifacts

- `assets/web/drift_recovery.png` (review) and `assets/paper/drift_recovery.png` (publication) - recovery behavior under drift
- `assets/web/dashboard_multimode.png`
- `assets/web/dashboard_topology.png`
- `assets/web/dashboard_digital_twin.png`
- `assets/gkp_proxy.png` and `assets/web/gkp_proxy.png` - appendix-only intuition

## Reproduce

- Core reviewer flow:
  - `python scripts/onboard_from_raw.py --data-path data/raw --profile both --target advisor --verify`
  - `python scripts/build_assets_profiles.py --profile both --target advisor`
  - `python scripts/verify_assets_profiles.py --profile both --target advisor`

- `make test` (or `python -m pytest -q`)
- `make lint` (or `python -m ruff check scripts/export_golden_vectors.py tests/test_control_sweeps.py tests/test_hdl_feedforward.py tests/test_drift_automation.py`)
- `python scripts/generate_control_sweeps.py`
- `python scripts/export_golden_vectors.py --output-dir hdl/vectors --count 128 --seed 13 --manifest hdl/vectors/contract.json`
- `make -C hdl sim` (requires Icarus Verilog: `iverilog`, `vvp`)
- `python scripts/simulate_24h_drift.py`

## Claim Mapping

- Control constraints: `assets/web/sweep_latency.png`, `assets/web/sweep_quantization.png`, `python scripts/generate_control_sweeps.py`
- Nonlinear feedforward / FPGA: `hdl/feedforward_lut.sv`, `hdl/tb_feedforward_lut.sv`, `hdl/Makefile`, `scripts/export_golden_vectors.py`, `hdl/vectors/contract.json`, `waves.vcd`
- Digital-twin/control co-design: `scripts/simulate_24h_drift.py`, `quantum_optical_bus.control`
- Drift / automation: `assets/web/drift_recovery.png`, `scripts/simulate_24h_drift.py`, `tests/test_drift_automation.py`

## Roadmap / appendix note

- Appendix visuals are excluded from MVP acceptance: `gkp_proxy`, scenario/advanced gallery GIFs, and other full-target-only polish outputs.
- `docs/APPENDIX_GKP.md` is retained only for historical intuition and should not be used as primary evidence.


