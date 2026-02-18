# OQC World-Modeling Control Co-design MVP — Project Specification

## 0) One-line goal
Build a reproducible, evidence-first portfolio that connects:
(1) world modeling (data assimilation → model update)
→ (2) control co-design (latency/filter/quantization)
→ (3) deployable artifacts (HDL feedforward, coefficients)
for time-domain photonic quantum computing.

## 1) Why this exists (PI-facing intent)
This repo must produce concrete artifacts that support four claims:
A) Fault tolerance / logical qubit / GKP language (toy but correct).
B) Nonlinear feedforward as a primary engineering target (HDL-ready).
C) Long-term stability & automation (24/365-style operation mindset).
D) World modeling loop: data → system ID → model update → deployment artifacts.

## 2) Scope (MVP)
We will implement a *minimal but defensible* stack:

### A) Digital Twin (Gaussian-first)
- Minimal time-bin / loop-like simulation (or topology abstraction if full loop is too heavy).
- Simple Gaussian gate sequence + measurement stream output.
- Deterministic seeds, reproducible notebook runs.

### B) Control Constraints Sweeps
Quantify how these constraints degrade metrics:
- Latency (bins/cycles)
- Filtering (simple FIR/IIR model, or group-delay approximation)
- Fixed-point quantization (bit width, rounding/saturation)

Outputs: stable plots under `assets/` + a single script to regenerate them.

### C) FPGA/HDL Nonlinear Feedforward Reference
Implement a minimal HDL block:
- Input: homodyne stream sample (fixed-point)
- Core: LUT-based nonlinear mapping (or piecewise polynomial)
- Output: drive value (fixed-point)
- Explicit pipeline stages so cycle latency is measurable

Verification: VCD waveform + golden-vector checks.

### D) Fault-tolerance / GKP Toy Model (logical proxy)
Implement a minimal, explainable GKP-EC toy:
- finite squeezing / shift-noise proxy
- output: logical error proxy (or syndrome residual)
- sweep: noise vs proxy (and optionally control quantization/latency effect)

Output: at least one plot under `assets/`.

### E) Stability / Automation (simulation-based)
Implement a drift + auto-calibration loop:
- drift generator (phase / loss / effective squeezing drift)
- estimator (EMA/Kalman-like minimal)
- controller update (recompute coefficients)
- demonstrate recovery over long horizon (simulated “24h”)

Output: plot under `assets/` + reproducible script.

## 3) Non-goals (to keep it shippable)
- No claims of matching a specific lab’s proprietary setup.
- No full-scale cluster-state / full MBQC compiler required for MVP.
- No real FPGA board integration in MVP (HDL sim + bit-accurate checks only).
- No unstable “demo-only” code; everything must be reproducible.

## 4) Repo layout (target)
- docs/
  - PROJECT_SPEC.md (this file)
  - EVIDENCE_PACK.md (generated summary of artifacts)
  - ARCHITECTURE.md (1–2 pages, diagrams ok)
- assets/ (generated artifacts, committed)
  - sweep_latency.png
  - sweep_quantization.png
  - gkp_proxy.png
  - drift_recovery.png
- notebooks/ (repro notebooks)
  - 01_tdm_minimal.ipynb
  - 02_control_sweeps_explainer.ipynb (optional)
  - 03_hdl_latency_demo.ipynb (optional)
  - 04_gkp_ec_toy.ipynb
  - 05_drift_auto_calibration.ipynb (optional)
- scripts/ (single-command reproducibility)
  - generate_control_sweeps.py
  - export_golden_vectors.py
  - simulate_24h_drift.py
  - run_gkp_sweep.py
- hdl/
  - feedforward_lut.sv
  - tb_feedforward_lut.sv
  - Makefile (make sim -> VCD)
  - waves.vcd (generated; can be gitignored, but provide command)
- src/ (existing package; extend instead of duplicating)

## 5) Definition of Done (DoD)
A change is “done” only if:
1) `make test` passes (or `pytest` if makefile does not exist).
2) `make lint` passes (or equivalent formatter/linter command).
3) `python scripts/generate_control_sweeps.py` regenerates `assets/sweep_*.png`.
4) `make -C hdl sim` produces a VCD waveform and prints measured cycle latency.
5) Every notebook runs top-to-bottom with no manual steps (document the command).
6) `docs/EVIDENCE_PACK.md` lists:
   - what was built
   - how to reproduce
   - which SOP/CV claim each artifact supports

## 6) Evidence mapping (SOP/CV-ready)
- GKP / fault tolerance:
  - assets/gkp_proxy.png, notebooks/04_gkp_ec_toy.ipynb, scripts/run_gkp_sweep.py
- Nonlinear feedforward / FPGA:
  - hdl/feedforward_lut.sv, hdl/tb_feedforward_lut.sv, hdl/Makefile, VCD proof
- Stability / automation:
  - assets/drift_recovery.png, scripts/simulate_24h_drift.py
- World modeling loop:
  - docs/ARCHITECTURE.md (data→ID→update→deploy diagram)
  - scripts/export_golden_vectors.py (deployment artifact generation)
  - (optional) a small example dataset under data/examples/

## 7) Reproduce commands (must remain valid)
- Python:
  - pip install -e .
  - make test
  - python scripts/generate_control_sweeps.py
- HDL:
  - make -C hdl sim
