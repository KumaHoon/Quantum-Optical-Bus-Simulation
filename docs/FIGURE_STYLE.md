# Figure Style and Contract (web / paper)

## Purpose

This document is the single source of truth for figure appearance, export policy, and reproducibility requirements for repository artifacts.

It is aligned with the review path in `README.md` and is used by the contract-driven verification flow.

## Scope

Applies to:

- `assets/web/**/*.png`
- `assets/web/**/*.gif`
- `assets/paper/**/*.png`
- `assets/paper/**/*.pdf`
- Figures produced by scripts under `scripts/`

## Profiles

### web

- Background: light (white)
- Primary format: PNG
- Optional GIF: only when sequence behavior is essential
- Canvas: `1600 x 1000` px
- DPI: `150`
- Primary use: README and quick-review

### paper

- Background: white
- Formats: PNG + PDF
- Canvas: `7.0 x 4.5` in
- DPI: `300`
- Primary use: publication-oriented outputs

## Global styling rules

### One figure = one claim

- Each figure should support one primary claim.
- If multiple panels are needed, keep each panel understandable by itself and label as `a`, `b`, `c`.

### Typography hierarchy (minimums)

For web figures:

- Title: `16-20` pt
- Axis labels: `14-16` pt
- Tick labels: `12-14` pt
- Legend: `12-14` pt
- Annotation text: `>= 12` pt

For paper figures:

- Title: `10-12` pt
- Axis labels: `9-10` pt
- Tick labels: `8-9` pt
- Legend: `8-9` pt
- Annotation text: `>= 8` pt

### Line and marker defaults

- Line width: `2.0` (web), `1.5` (paper)
- Marker size: `6` (web), `4` (paper)
- Avoid dense markers on continuous curves

### Color and accessibility

- Do not use rainbow colormaps.
- Do not rely on color alone for critical comparisons.
- Use redundant channels: line style, marker, label text, and annotation.
- Use perceptually uniform heatmaps (e.g., viridis-like style family).
- Ensure grayscale-safe readability.

### Axes and labels

- Every axis requires a label.
- Unit brackets are required for unitful quantities (for example `[mW]`, `[dB]`, `[rad]`, `[vac=0.5]`).
- Legends must not occlude data.
- Keep labels, ticks, titles, and annotations outside important data ranges.

### Layout and clipping

- Use `constrained_layout=True` or equivalent.
- No title/axis/annotation clipping.
- Preserve readable spacing between neighboring subplots and text blocks.

## Export rules

### Destination

- Web: `assets/web/`
- Paper: `assets/paper/`

### Naming

- Figure image: `assets/<profile>/<figure_id>.png`
- Figure PDF (paper): `assets/paper/<figure_id>.pdf`
- GIF: `assets/web/<figure_id>.gif` (sequence figures only)
- Data companion: `assets/<profile>/data/<figure_id>.npz`
- Metadata companion: `assets/<profile>/meta/<figure_id>.meta.json`

### Required metadata fields

Each generated figure must include at least:

- `figure_id`
- `profile`
- `created_at_utc`
- `generator_script`, `generator_args`
- `seed` (nullable if not used)
- `canvas_px` (web) and/or `canvas_in` (paper)
- `dpi`
- `labels` (`title`, `xlabel`, `ylabel`, optional `y2label`)
- `units` (canonical variable-to-unit map)
- `notes` (1-2 lines on main claim)
- `git_commit` (if available)
- `python_version`

## Determinism and verification expectations

- Deterministic output for fixed RNG seeds.
- Stable output paths for identical inputs.
- Verification should read numeric companions and metadata, not pixel hashes.
- Validation checks use `docs/FIGURE_CONTRACT.yaml` and `docs/VALIDATION_FUNCTIONS.md`.

## Reviewer flow (recommended)

```bash
python scripts/build_assets_profiles.py --profile both --target advisor
python scripts/build_assets_profiles.py --profile both --target advisor --verify
```

- Web outputs are the default reviewer path.
- Paper outputs are used after web approval.

## References

- `docs/figure_checklist.md` (human review rubric)
- `docs/FIGURE_CONTRACT.yaml` (required figures and expected outputs)
- `docs/UNITS_AND_CONVENTIONS.md` (labels, units, and conventions)
- `docs/VALIDATION_FUNCTIONS.md` (validator names and semantics)
