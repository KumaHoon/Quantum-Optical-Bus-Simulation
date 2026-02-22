---
name: oqc-mvp
description: Implement the OQC Digital-Twin Control Co-design MVP in this repo. Use ONLY when working on deliverables defined in docs/PROJECT_SPEC.md (digital twin, control sweeps, HDL feedforward, drift automation). Do NOT use for unrelated refactors.
---

## When to use
Use this skill when the task is explicitly part of docs/PROJECT_SPEC.md.

## Operating rules (must follow)
1) Always read docs/PROJECT_SPEC.md first and follow its DoD.
2) Prefer small, verifiable commits.
3) Do not add heavy dependencies unless required for a deliverable.
4) After multi-file changes, run (or request permission if needed):
   - make test (or pytest)
   - make lint
5) Every deliverable must have a reproducible command under scripts/ or Makefile.
6) Never claim a feature is done without producing its artifact under assets/ or hdl/ and updating docs/EVIDENCE_PACK.md.

## Standard workflow
1) Explore: locate the best integration points in src/ (avoid duplication).
2) Implement: add minimal code to satisfy a single deliverable.
3) Verify: run required commands and confirm artifacts exist.
4) Document: update docs/EVIDENCE_PACK.md with reproduction steps.

## Required commands checklist
- Python: make test, make lint, python scripts/generate_control_sweeps.py
- HDL: make -C hdl sim
