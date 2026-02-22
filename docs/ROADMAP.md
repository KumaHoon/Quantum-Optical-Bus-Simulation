# ROADMAP

This roadmap defines phased milestones for the Quantum Optical Bus project.

## Review scope note

- Core reviewer evidence remains the README-linked path only.
- Core path is fixed to the 3-item MVP evidence trio shown in `README.md`:
  - `assets/web/calibration_demo.gif`
  - `assets/web/dashboard_decoherence.png`
  - `assets/web/sweep_latency.png` or `assets/web/sweep_quantization.png`
- Route extensions (`APPENDIX_GKP.md`, advanced gallery artifacts, and future fault-tolerance routes) stay non-blocking for this cycle.

## P0 - Bootstrap and Reliability
Scope:

- Keep package install/test flow stable across CI platforms.
- Maintain compatibility patches in `compat.py`.
- Provide a minimal task runner and contributor instructions.

Acceptance criteria:

- `python -m pip install -e ".[test]"` succeeds.
- `python -m pytest -q` passes.
- `make test` target exists and maps to pytest.
- Dashboard launch command is documented.

## P1 - Calibration Model Hardening
Scope:

- Strengthen `interface.py` input validation and edge-case handling.
- Ensure intrinsic vs observed squeezing semantics are explicit in docs/tests.
- Add tests for loss monotonicity and parameter boundaries.

Acceptance criteria:

- Unit tests cover zero/positive/high power paths and loss mapping.
- Intrinsic squeezing remains loss-independent in tests.
- Observed squeezing decreases monotonically with increasing loss.

## P2 - Quantum/Visualization Quality
Scope:

- Improve `quantum.py` result ergonomics and numerical robustness.
- Reduce duplicate calculations in dashboard rendering.
- Validate plotted metrics against backend state outputs.

Acceptance criteria:

- Shared simulation helper is used consistently by app and tests.
- No regression in Wigner/variance/mean-photon outputs for baseline scenarios.
- App remains interactive at default slider settings.

## P3 - Hardware-Calibration Integration
Scope:

- Replace placeholder calibration constant with data-informed calibration inputs.
- Add a repeatable pipeline for mode-overlap-derived parameters.
- Document assumptions and validation procedure.

Acceptance criteria:

- Calibration parameter source is traceable and versioned.
- Integration tests confirm hardware-to-quantum parameter handoff.
- README includes reproducible calibration workflow.
