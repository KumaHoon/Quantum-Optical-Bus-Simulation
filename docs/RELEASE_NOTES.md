# Release Notes

## vNext (Phase 10 regeneration run)

### What changed
- Regenerated all dashboard and evidence visual assets to align with the latest styling/layout updates.
- Rebuilt the animated live-demo and gallery artifacts for current code state:
  - `assets/dashboard_vacuum.png`
  - `assets/dashboard_calibration.png`
  - `assets/dashboard_decoherence.png`
  - `assets/dashboard_multimode.png`
  - `assets/dashboard_topology.png`
  - `assets/dashboard_digital_twin.png`
  - `assets/scenario_gallery.gif`
  - `assets/advanced_gallery.gif`
  - `assets/advanced_evidence.gif`
  - `assets/calibration_demo.gif`
- Rebuilt HIL infographic artifacts:
  - `docs/figures/hil_expansion.png`
  - `docs/figures/hil_expansion.pdf`
- Kept label/axis-unit consistency and unclipped rendering updates from prior phases.
- i18n docs and reference/architecture updates remain synchronized with rendered assets.

### Reproduction
From repository root:
1. `python scripts/generate_dashboard_gallery.py`
2. `python scripts/generate_advanced_dashboard_gallery.py`
3. `python scripts/generate_calibration_demo.py`
4. `python scripts/generate_scenario_gallery_gif.py`
5. `python scripts/generate_advanced_gallery_gif.py`
6. `python scripts/generate_advanced_evidence_gif.py`
7. `python scripts/generate_hil_infographic.py`

### Quality gates
- `make lint` / `make test` were attempted but `make` is not installed in this environment.
- Windows-equivalent checks used:
  - `python -m ruff check .`
  - `python -m ruff format --check .`
  - `python -m compileall src tests`
  - `python -m pytest -q`

### Gate results
- `python -m ruff check .` passed.
- `python -m ruff format --check .` passed after formatting.
- `python -m compileall src tests` passed.
- `python -m pytest -q` passed: **38 passed, 0 failed**.

### Notes
- `python scripts/generate_calibration_demo.py` required a longer runtime window due GIF frame generation and optimization. No functional changes were made during this run.
