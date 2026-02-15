# AGENTS.md

## Purpose
Concise contributor instructions for agentic development in this repository.

## Setup
```bash
python -m pip install -e ".[test]"
```

## Core Commands
```bash
make test
make app
make lint
```

If `make` is unavailable (common on Windows), use:
```bash
python -m pytest -q
streamlit run src/quantum_optical_bus/calibration_app.py
python -m compileall src tests
```

## Definition of Done
- Tests pass locally: `python -m pytest -q`
- CI workflow remains green on Ubuntu/macOS/Windows
- Existing scripts keep working:
  - `python scripts/generate_dashboard_gallery.py`
  - `python scripts/generate_calibration_demo.py`

## Guardrails
- Keep public APIs stable unless explicitly requested.
- Prefer small, reviewable changes.
- Fix root causes; do not suppress errors.
