---
name: oqc-evidence
description: Summarize generated artifacts into docs/EVIDENCE_PACK.md and draft SOP/CV-ready documentation.
---

## When to use
Use this skill after you generate new repository artifacts (`assets/`, `scripts/`, or `hdl/`).

## What to produce
1) docs/EVIDENCE_PACK.md
   - One-sentence claim (must be supported by repo artifacts)
   - Artifact list with filenames
   - Reproduce commands (3-10 lines)
   - Mapping to implemented capability claims (digital-twin/control co-design, nonlinear FF, stability/automation, FPGA contract)
2) docs/SOP_SNIPPETS.md
   - 3-6 SOP-ready sentences (no overclaim)
3) docs/CV_PROJECT_BULLETS.md
   - 4-8 CV bullets with measurable outputs (files, scripts, latency, plots)

## Figure design memory check (effective scientific figures)
- Before finalizing or accepting new figures, re-check the `docs/figure_checklist.md` rules:
  - Canonical guidance source: [Designing effective scientific figures](https://bioinformatics-core-shared-training.github.io/effective-figure-design/DesigningEffectiveScientificFigures_Zabala_afternoon_v00.pdf)
  - Hard visual contract: `docs/figure_style_contract.md`
  - Automate this check where possible:
    `python scripts/verify_figure_style_contract.py --profile web --target advisor`
  - one figure, one message
  - all labels/legend/axis text are readable without external context
  - minimal ink; no decorative clutter that does not change interpretation
  - axis units and tick logic are consistent across related panels
  - avoid color-only coding; include redundant encoding when needed
  - maintain readability after reduction (line/marker visibility)
  - roadmap-only visuals are explicitly excluded from MVP acceptance
- Prefer `assets/web` outputs for initial README validation and use `assets/paper` only for final polishing.
- For onboarding updates, ensure README path is followed (`onboard_from_raw.py` / `build_assets_profiles.py` with `--target advisor`) before claiming completion.

## OQC Evidence operating principles

- Document priority:
  - Core docs define review scope: `README.md`, `docs/PROJECT_SPEC.md`, `docs/figure_checklist.md`, `docs/EVIDENCE_PACK.md`, `docs/ARCHITECTURE.md`, `docs/data_schema.md`.
  - Roadmap/Appendix docs are non-gating: `docs/ROADMAP.md`, `docs/REFERENCES.md`.
  - Archive docs are optional context: `docs/AUDIT_REPORT.md`, `docs/RELEASE_NOTES.md`, `docs/APPENDIX_GKP.md`.
- Readme-first route:
  - Keep a 30s/3-minute flow clear from README: one-line goal -> run path -> live demo -> implemented-vs-roadmap -> onboarding -> architecture -> figure policy.
  - Any claim outside this path should be tagged roadmap or appendix.
- Evidence rule:
  - Apply `Claim -> Artifact -> Verification` for all reportable capabilities in `docs/EVIDENCE_PACK.md`.
  - For MVP, use review commands aligned to `--target advisor`.
- Figure rule:
  - One figure carries one message.
  - Labels, legends, and units must be interpretable without context lookup.
  - Axis/legend logic must match panel intent; avoid color-only communication.
- Roadmap-only visuals (e.g., `gkp_proxy`, `scenario_gallery`, `advanced_gallery`, `advanced_evidence`) are explicitly excluded from MVP acceptance.
- Encoding and command checks:
  - Keep UTF-8 clean outputs; remove mojibake patterns (`U+00E2`, `U+0080`/`U+0093` combos) before finishing docs.
  - Keep profile consistency explicit: `assets/web` for review and `assets/paper` for publication.
  - Standard onboarding flow: `onboard_from_raw.py` -> `build_assets_profiles.py` -> `verify_assets_profiles.py`.

## Rules
- Only describe what exists in the repo.
- If something is missing, add a TODO section instead of guessing.

## Documentation quality gate (run before closing docs changes)
- Run this checklist from repo root after any reviewer-facing doc update:
  - Figure style contract automation:
    `python scripts/verify_figure_style_contract.py --profile both --target advisor`
  - ASCII/UTF-8 sanity for review path:
    `rg -n "â|â€|â€“|âœ˜|Ã" README.md docs .agents/skills/oqc-evidence/SKILL.md`
  - Korean text scan for consistency with English policy:
    `rg -n "[\\uAC00-\\uD7A3]" README.md docs .agents/skills/oqc-evidence/SKILL.md`
  - Core onboarding/docs commands must match the current standard:
    `rg -n "onboard_from_raw.py|build_assets_profiles.py|verify_assets_profiles.py" README.md docs/PROJECT_SPEC.md docs/figure_checklist.md docs/EVIDENCE_PACK.md`
  - Profile usage stays explicit:
    `rg -n "assets/web|assets/paper" README.md docs/PROJECT_SPEC.md docs/figure_checklist.md docs/EVIDENCE_PACK.md`
  - MVP claims must be in core docs only:
    `rg -n "gkp_proxy|scenario_gallery|advanced_gallery|advanced_evidence" README.md docs/PROJECT_SPEC.md docs/figure_checklist.md`

- Failure policy:
  - Any mojibake hit, Hangul hit, or path mismatch must be fixed before merging.
  - If a roadmap item is found in core claim paths, move it to appendix/roadmap section and re-run this gate.


