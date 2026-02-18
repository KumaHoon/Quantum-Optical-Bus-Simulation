---
name: oqc-evidence
description: Summarize repo artifacts into docs/EVIDENCE_PACK.md and draft SOP/CV-ready bullet points. Use only after new artifacts are generated (assets/, hdl/).
---

## When to use
Use this skill after you generate new assets/ plots, notebooks, or HDL artifacts.

## What to produce
1) docs/EVIDENCE_PACK.md
   - One-sentence claim (must be supported by repo artifacts)
   - Artifact list with filenames
   - Reproduce commands (3–10 lines)
   - Mapping to 4 capability claims (GKP, nonlinear FF, stability, world model)
2) docs/SOP_SNIPPETS.md
   - 3–6 SOP-ready sentences (no overclaim)
3) docs/CV_PROJECT_BULLETS.md
   - 4–8 CV bullets with measurable outputs (files, scripts, latency, plots)

## Rules
- Only describe what exists in the repo.
- If something is missing, add a TODO section instead of guessing.
