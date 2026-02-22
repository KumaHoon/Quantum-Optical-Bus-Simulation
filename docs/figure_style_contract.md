# Figure Style Contract (web / paper)

> Deprecated alias: the canonical documentation is now
> `docs/FIGURE_STYLE.md` + `docs/FIGURE_CONTRACT.yaml`.

## Purpose

This document is the **single source of truth** for figure appearance, export policy, and reproducible evidence-generation constraints.
It is consumed by automation in `scripts/verify_figure_style_contract.py`.

## Scope

- `assets/web/**/*.png`, `assets/web/**/*.gif`
- `assets/paper/**/*.png`, `assets/paper/**/*.pdf`
- Matplotlib/Pillow figures produced by scripts in `scripts/`

## Profiles

### `web`
- Background: light (default dark-mode compatible white)
- Output format: PNG (primary), GIF only for essential multi-frame behavior
- Canvas: **1600 × 1000 px**
- DPI: **150**
- Primary use: README / quick review

### `paper`
- Background: pure white
- Output format: PNG + PDF
- Canvas: **7.0 in × 4.5 in** (`2100 × 1350 px @ 300 dpi`)
- DPI: **300**
- Primary use: publication and slide artifacts

## Global style rules

### One figure = one claim
- Each artifact must be understandable without requiring another panel.
- If multiple panels are necessary, label panels (`A`, `B`, `C`) and keep each panel legible alone.

### Typography
- Web: title 16–20, axis label 14–16, tick label 12–14, legend 12–14, annotation ≥12
- Paper: title 10–12, axis label 9–10, tick label 8–9, legend 8–9, annotation ≥8

### Line / marker defaults
- Line width: 2.0 (web), 1.5 (paper)
- Marker size: 6 (web), 4 (paper)

### Color and accessibility
- No rainbow colormaps.
- No critical message should rely on color alone.
- Use at least one redundant channel: line style, marker, text, or annotation.
- Heatmaps should use perceptually uniform options (e.g., viridis-like).
- Grayscale readability should remain meaningful.

### Legends and labels
- Legends must avoid data occlusion.
- Every axis must include:
  - label
  - unit bracket when unitful
- Unitless quantities should be explicitly annotated (e.g., `[unitless]`).

### Layout and clipping
- No clipped text.
- Use `constrained_layout=True` or equivalent.
- Prevent title/label overlap with neighboring panels.

## Export and reproducibility rules

### Output destinations
- Web: `assets/web/`
- Paper: `assets/paper/`

### Naming
- Single figures: `assets/<profile>/<figure_id>.png` (and `.pdf` for paper)
- Data companion: `assets/<profile>/data/<figure_id>.npz`
- Metadata: `assets/<profile>/meta/<figure_id>.meta.json`

### Required metadata per figure
Each PNG/GIF must emit JSON metadata containing:
- `figure_id`, `profile`, `created_at_utc`
- `generator_script`, `generator_args`
- `seed` if used
- `canvas_px`, `dpi`
- `labels` (`title`, `xlabel`, `ylabel`, optional secondary labels)
- `units` and `notes` (1–2 lines, claim-oriented)
- `git_commit` (when available), `python_version`

### Determinism
- Fixed seeds for all stochastic generation steps.
- Stable output paths.
- Verification is performed from PNG/PDF **and** metadata/`.npz` companions.

## Minimal acceptance checks
Verification fails when any of the following is missing or invalid:
- required figure not found for the selected target
- required DPI or canvas mismatch
- missing required metadata
- missing axis title/labels/units
- missing expected `.pdf` companion for paper PNGs
- text clipping or layout truncation (best-effort checks)

## Reference
- `docs/figure_checklist.md` (review rubric)
- `scripts/verify_figure_style_contract.py`
