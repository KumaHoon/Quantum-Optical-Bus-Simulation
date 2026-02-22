# Loop-based OQC Digital-Twin + Control Co-design MVP

## 0) One-line goal
This repository targets loop-based / time-domain-multiplexed optical quantum workflows, where squeezed-state generation and homodyne measurement are coupled with model identification and control updates. The closed-loop path (measurement -> estimation -> control) is represented explicitly as a reproducible digital-twin + control pipeline.

## 1) Why this exists (technical intent)
- **HDL-ready nonlinear feedforward** (control-pipeline readiness)
- **24/365-style stability automation** (drift detection and retuning)
- **Quantify how latency/quantization constraints affect observable metrics under loop-like conditions.**
- MVP evidence scope: only README-linked artifacts and claims are acceptance-critical; roadmap-only visuals are documentation-only.
- README-facing MVP evidence now starts with:
  - `assets/web/calibration_demo.gif`
  - `assets/web/dashboard_decoherence.png`
  - `assets/web/sweep_latency.png` (or `assets/web/sweep_quantization.png`)

## 2) Scope (MVP)
### A) Digital Twin (Gaussian-first)
- 4-mode time-bin, loop-like simulation (laptop-runnable, deterministic seeds)
- Outputs include state-quality verification proxies derived from covariance (e.g., nullifier-like metrics / correlation summaries) to support scaling-relevant evaluation.

### B) Control Constraints Sweeps
- Measure metric sensitivity to latency, filtering, delay, and quantization.
- Baseline sweep settings:
  - latency sweep: `0..10` cycles
  - quantization sweep: bit widths `[2, 3, 4, 5, 6, 8, 10, 12]`
  - synthetic drift generation + estimator/control pipeline in `control.py` and `simulate_24h_drift.py`
- These values are selected as baseline choices under FPGA latency/resource constraints and validated with sensitivity sweeps.
- The drift model is a reduced-order profile covering observed platform scales (minutes-to-hours) before hardware-identified updates.
- Reproducible outputs under `assets/<profile>/` via `scripts/onboard_from_raw.py`/`scripts/build_assets_profiles.py` one-flow, with `--target advisor`.
  - reviewer-facing: `assets/web`
  - publication-ready: `assets/paper`
- Pipeline path: `fit -> simulated drift/plant -> feedback control` is executed as `measurement -> estimation -> control` in the closed-loop utility.

### C) FPGA/HDL Nonlinear Feedforward
- Fixed-point homodyne -> LUT/polynomial nonlinearity -> fixed-point control out
- Explicit pipeline stages for measurable latency
- Contracted FPGA interface:
  - Input: signed Q1.15 sample stream, `in_valid`, `clk`, `rst_n`
  - Output: signed Q1.15 control stream, `out_valid`
  - Reference RTL latency: 2 cycles (`PIPELINE_LATENCY`)
  - Golden vectors are regenerated reproducibly by `scripts/export_golden_vectors.py --manifest ...`
- Verified by VCD + manifest-backed golden-vector checks

### D) Stability / Automation
- 24h drift synthesis + EMA estimator (`alpha = 0.22` in the default profile) + adaptive coefficient update
- Drift is evaluated on deterministic synthetic trajectories to test closed-loop recovery over platform-relevant time scales.
- Output: `assets/<profile>/drift_recovery.png` (`assets/web/` for review profile, `assets/paper/` for publication profile)

### E) Data-to-control (roadmap)
- `data/raw` -> `scripts/fit_lab_data.py` -> `scripts/build_assets_profiles.py`
- standard onboarding path:
  - `python scripts/onboard_from_raw.py --data-path data/raw --profile both --target advisor --verify`

## 3) Non-goals
- No full MBQC compiler, no board-level FPGA bring-up, and no demonstration-only code.
- No full GKP encoder/decoder stack; no fault-tolerance encoding claims are part of MVP evidence.

## 4) Repo layout (target)
- `docs/`: `PROJECT_SPEC.md`, `EVIDENCE_PACK.md`, `ARCHITECTURE.md`
- `data/raw/`: experiment CSV drops
- `assets/`: profile-based generated artifacts
  - `assets/web/...` for review artifacts (`sweep_latency.png`, `sweep_quantization.png`, `drift_recovery.png`)
  - `assets/paper/...` for publication artifacts (same filenames in publication-safe rendering)
- `scripts/`: `generate_control_sweeps.py`, `export_golden_vectors.py`, `simulate_24h_drift.py`
- `hdl/`: `feedforward_lut.sv`, `tb_feedforward_lut.sv`, `Makefile`, `waves.vcd`
- `src/`: extend existing package

## 5) Definition of done
1. `make test` / `pytest` passes
2. `make lint` passes
3. `python scripts/generate_control_sweeps.py` regenerates `assets/web/sweep_*.png` (review profile) and `assets/paper/sweep_*.png` (publication profile)
4. `make -C hdl sim` generates VCD and prints cycle latency
5. `python scripts/onboard_from_raw.py --profile both --target advisor --verify` aligns README core artifacts
6. Figure scripts regenerate assets reproducibly
7. `docs/EVIDENCE_PACK.md` includes Claim / Evidence / Limitation per artifact

## 6) Evidence mapping
| Claim | Evidence | Limitation |
| --- | --- | --- |
| Nonlinear feedforward / FPGA | `hdl/feedforward_lut.sv`, `hdl/tb_feedforward_lut.sv`, `hdl/Makefile`, `scripts/export_golden_vectors.py` (Q-format + manifest), `hdl/vectors/contract.json`, VCD waveform | HDL-only validation; no board-level bring-up |
| Stability / automation | `assets/<profile>/drift_recovery.png`, `scripts/simulate_24h_drift.py` | Synthetic drift model only |

## 7) Reproduce commands
- `pip install -e .`
- `make test`
- `python scripts/generate_control_sweeps.py`
- `make -C hdl sim`


