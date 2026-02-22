# Release Notes

## Scope note

- Archive note: this file is historical and non-authoritative for MVP acceptance.
- Legacy/legacy-style gallery and evidence polish scripts are preserved for reproducibility, not reviewer gating.

## vNext (Phase 10 regeneration run)

### What changed
- Regenerated dashboard and evidence visual artifacts.
- Rebuilt:
  - `assets/web/dashboard_vacuum.png`
  - `assets/web/dashboard_calibration.png`
  - `assets/web/dashboard_decoherence.png`
  - `assets/web/dashboard_multimode.png`
  - `assets/web/dashboard_topology.png`
  - `assets/web/dashboard_digital_twin.png`
  - `assets/paper/dashboard_*.png` (for `--profile both`)
- `assets/web/calibration_demo.gif`
- `assets/paper/calibration_demo.gif` (for `--profile both` publication output)
- `assets/scenario_gallery.gif` (roadmap / optional)
- `assets/advanced_gallery.gif` (roadmap / optional)
- `assets/advanced_evidence.gif` (roadmap / optional)
- Rebuilt HIL infographic artifacts:
  - `docs/figures/hil_expansion.png`
  - `docs/figures/hil_expansion.pdf`
- Kept label/axis-unit consistency and unclipped rendering updates from prior phases.
- i18n docs and reference/architecture updates remain synchronized with rendered assets.

### Reproduction
- `python scripts/generate_dashboard_gallery.py`
- `python scripts/generate_advanced_dashboard_gallery.py`
- `python scripts/generate_calibration_demo.py`
- `python scripts/generate_scenario_gallery_gif.py`
- `python scripts/generate_advanced_gallery_gif.py`
- `python scripts/generate_advanced_evidence_gif.py`
- `python scripts/generate_hil_infographic.py`

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
- `python scripts/generate_calibration_demo.py` required longer runtime due to GIF frame generation and optimization.
- This file is historical context and does not redefine MVP acceptance boundary.
