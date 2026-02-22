# Figure Checklist for reviewer use (`dashboard_*.png` + sweep PNGs)

## Standard workflow (standard routine)
- [ ] Generate all web/paper assets with one command:
  `python scripts/onboard_from_raw.py --profile both --target advisor --verify`
- [ ] Or, if no raw CSV is available yet:
  `python scripts/build_assets_profiles.py --profile both --target advisor`
- [ ] Run verification immediately after generation:
  `python scripts/verify_assets_profiles.py --profile both --target advisor`
- [ ] Run the style contract verifier after generation:
  `python scripts/verify_figure_style_contract.py --profile both --target advisor`
- [ ] Or run the checklist driver directly (generate + verify + audit):
  `python scripts/check_figure_checklist.py --profile both --target advisor`
- [ ] For profile-only refresh, use:
  - `python scripts/build_assets_profiles.py --profile web`
  - `python scripts/build_assets_profiles.py --profile paper`
- [ ] Keep acceptance checks to core README-linked artifacts; treat all roadmap/appendix entries as optional.
- [ ] Keep the first-pass acceptance scope to the 3-item MVP trio (calibration demo, decoherence, one control-sensitivity figure).

## README-oriented figure design QA (effective-figure-first)

Use this as the default review layer for figures shown or linked from README.

Hard constraints are defined in:
- `docs/FIGURE_STYLE.md` (style contract + hard checks)
- `docs/FIGURE_CONTRACT.yaml` (required-figure map and scope)
- `scripts/verify_figure_style_contract.py` (automated enforcement)

### Core rules (5-second read intent)
- [ ] One figure = one claim/message. Remove secondary narratives from the same panel.
- [ ] All labels (axes, legend, panel notes) are fully understandable without looking elsewhere.
- [ ] Data is shown with minimal ink:
  - no decorative effects that do not change interpretation
  - no cluttered grid/frames/markers
- [ ] Quantities are faithful to the data:
  - start/stop axes are tight and non-misleading
  - units and ticks are consistent across related panels
  - legends encode line/marker meaning directly
- [ ] Accessible comparison rules:
  - avoid relying on color alone
  - keep lines/markers thick enough to read when viewed at reduced size
  - keep panel titles brief and descriptive

Hard constraints covered by automation:
- [ ] Contract canvas and DPI match target profile.
- [ ] Required metadata exists (`meta/<figure>.meta.json`) and includes labels, units, notes, git/Python provenance.
- [ ] Paper outputs include `.pdf`; web/paper labels keep unit-safe axis notation.

### README-specific requirements
- [ ] For the first-pass README route, prefer web-readable versions in `assets/web`.
- [ ] Publication polishing (`assets/paper`) can apply a separate visual pass only after README acceptance.
- [ ] Every checklist item should be verifiable by opening the figure and its linked file path from README.

### Zabala (effective scientific figures) rubric

Reference: [Designing effective scientific figures](https://bioinformatics-core-shared-training.github.io/effective-figure-design/DesigningEffectiveScientificFigures_Zabala_afternoon_v00.pdf)

- [ ] Start with message-first structure: text and caption must state what is being compared and why (one claim per figure).
- [ ] Select the right chart for data type:
  - trend over time/range -> line
  - category comparison -> bar or dotchart
  - relationship/distribution -> scatter/hist/box/heatmap as needed
- [ ] Use marks/channels responsibly:
  - position/length for key quantitative comparisons
  - color/shape as secondary encoding, not the primary carrier for precision
  - avoid relying on hue alone; use labels/line style/marker for redundancy
- [ ] Make the comparison logic immediate:
  - readable axis titles with units
- [ ] Keep typography and layout hierarchical:
  - consistent font family and hierarchy (title > axis labels > tick labels)
  - balanced margins and aligned axes/grid for multi-panel figures
- [ ] Control optical noise:
  - no dense decoration, no redundant ornaments
  - no overloaded panel titles or duplicated explanatory text
- [ ] Validate ethics of color:
  - readable in grayscale-safe conditions
  - adequate contrast for annotation + line/marker separation

## Reviewer 30s / 3min route

### 0:00-0:30 (Rapid first-pass review)
- [ ] Read `What it is` and confirm the project target in one sentence.
- [ ] Confirm execution path (`Run it now`) is one-command lightweight.
- [ ] Confirm `Live demo` shows `intrinsic vs observed` and monotonic loss behavior.
- [ ] Confirm calibration demo and decoherence are included in the core visible set:
  - `assets/web/calibration_demo.gif`
  - `assets/web/dashboard_decoherence.png`
- [ ] Confirm one control-sensitivity artifact is present:
  - `assets/web/sweep_latency.png` or `assets/web/sweep_quantization.png`

### 0:30-1:00 (FPGA evidence verification pass)
- [ ] Run the three-line FPGA chain:
  - `python scripts/export_golden_vectors.py --output-dir hdl/vectors --count 128 --seed 13 --manifest hdl/vectors/contract.json`
  - `make -C hdl sim`
  - `python -c "import json; c=json.load(open('hdl/vectors/contract.json')); print(f\"Q-format={c['format']['format']} latency={c['format']['pipeline_latency_cycles']} cycles\")"`
- [ ] Pass criteria: `.mem` files + `contract.json` exist, VCD generated with no output mismatches, manifest shows `Q1.15` and 2-cycle latency.

### 1:00-3:00 (Quality review)
- [ ] Confirm implemented vs roadmap boundary in `Scope (Implemented vs Roadmap)`.
- [ ] Confirm onboarding path (`data/raw` -> `fit_lab_data.py` -> rebuild) in `Research-ready onboarding story`.
- [ ] Confirm loop-based/closed-loop dataflow in `Architecture`.
- [ ] Confirm figure-readiness of core evidence figures with clear labels/units:
  - `assets/web/calibration_demo.gif`
  - `assets/web/dashboard_decoherence.png`
  - `assets/web/sweep_latency.png` or `assets/web/sweep_quantization.png`
- [ ] Confirm README-oriented figure design QA checks are satisfied where relevant.
- [ ] Optional (extensions): confirm `dashboard_multimode.png`, `dashboard_topology.png`, `dashboard_digital_twin.png`, and `drift_recovery.png` only if cited as optional.

### 3-minute one-page execution script (recommended)
- [ ] Open this order and stop on failure:
  1. `What it is (30 seconds)`
  2. `Run it now`
  3. `Live demo`
  4. `Scope (Implemented vs Roadmap)`
  5. `Research-ready onboarding story`
  6. `Architecture`
  7. `Figure policy` (`dashboard_decoherence.png`, one `sweep_*.png`, `calibration_demo.gif`; optional dashboard extensions if cited)

Keep this section as the reviewer's first-pass checklist before detailed tuning.

## Core rule of thumb (review intent)
- Use this checklist for the first 30s~3min pass before detailed figure-tuning.
- Figure artifacts are accepted only when labels, axes, and units are immediately interpretable at a glance.

## README boundary summary
- [ ] MVP reviewer-visible set is exactly:
  - `assets/web/calibration_demo.gif`
  - `assets/web/dashboard_decoherence.png`
  - `assets/web/sweep_latency.png` or `assets/web/sweep_quantization.png`
- [ ] Keep `gkp_proxy`, gallery GIFs, and advanced evidence visuals out of MVP acceptance.

## General checklist (all figures)
- [ ] Confirm each figure has clear title, axis labels, and legend placement (no overlapping elements).
- [ ] Confirm no cropped edges on titles/labels.
- [ ] Confirm consistent figure scale (equal emphasis and not over-compressed).
- [ ] Confirm color/line weight/readability in grayscale-safe conditions.
- [ ] Confirm `set_aspect("equal")` and integer x-axis settings where lattice/grid logic applies.
- [ ] Confirm `Probability` y-axis is constrained to `[0, 1]`.

> "Single Figure" rule:
> Keep each figure self-contained in one panel so first-pass review does not depend on context panels.

## 1) `dashboard_vacuum.png`
- [ ] Ensure `Wigner function` + `Quadrature variance` layout is not compressed or overlapping.
- [ ] Add clear labels to Wigner right panel and colorbar without collision.
- [ ] `Quadrature variance` y-axis requirements:
  - [ ] Show 0.0~0.6 in `0.1` increments.
  - [ ] Use a readable metric label.
- [ ] Ensure X-axis and Y-axis aspect spacing are visually balanced.

## 2) `dashboard_calibration.png`
- [ ] For `Photon number distribution P(n)`, enforce:
  - [ ] `ylim` strictly within `[0, 1]`.
  - [ ] `P(n) >= 0`, and normalized scale sums near 1.
- [ ] Use sparse x ticks for `n` (`0,2,4,...`) and avoid clutter.
- [ ] Wigner panel should not overlap the `Photon distribution` region.
- [ ] No title duplication between `Wigner function` and `Photon distribution` blocks.

## 3) `dashboard_decoherence.png`
- [ ] Confirm x-axis/title logic: avoid mismatch in interpretation.
- [ ] Ensure label text states `Intrinsic (pre-loss) >= Observed (post-loss)` clearly.
- [ ] Keep summary text on a separate readable zone.
- [ ] Use `equal` Wigner aspect and fixed tick policy.

## 4) `dashboard_multimode.png`
- [ ] `Per-bin squeezing` legend must not overlay data.
- [ ] Set `Time-bin index` with integer ticks.
- [ ] Keep Wigner panel smaller and visually separated from neighboring subplots.
- [ ] Keep `Per-bin variances` metric label consistent (`Var(x)/Var(p)`).

## 5) `dashboard_topology.png`
- [ ] Set integer ticks for `Mode index`, `neighbor pair index`.
- [ ] `Corr(X)`, `Corr(P)` colorbar labels must be readable.
- [ ] Add consistent labels for `Cov(X)`, `Cov(P)`.
- [ ] Keep explanatory text inside the figure area.

## 6) `dashboard_digital_twin.png`
- [ ] Align fit-panel signal traces with labeled colors.
- [ ] Keep `Synthetic benchmark` note concise and explicit.
- [ ] For dual axis (`RMS residual`/`Retention`), separate tick formatting and labels.
- [ ] Prevent legend overlap with plotted lines.

## 7) `sweep_latency.png`
- [ ] Use `Latency steps` as x-axis with readable spacing.
- [ ] Use readable y tick intervals.
- [ ] Put legend and annotation outside crowded regions.

## 8) `sweep_quantization.png`
- [ ] Use readable quantizer x-axis ticks and labels.
- [ ] Show y-axis units (e.g., `(rad)`) with clear text.
- [ ] Keep legend concise and unambiguous.

## 9) `drift_recovery.png`
- [ ] Confirm 3-panel layout is visually separated (not crowded).
- [ ] Keep overlay labels clear (observed/expected).
- [ ] Dual-axis annotations should remain readable and color-separated.

## Post-script checks
- [ ] Ensure all figures are web-safe and not clipped.
- [ ] Store outputs by profile (`assets/web` for review, `assets/paper` for publication).
- [ ] GIFs used in review mode must contain more than 1 frame.

## README-only scope (review boundary)
- [ ] Keep acceptance criteria to figures and checks visible in README path.
- [ ] Treat roadmap-only visuals as optional when reviewing core repository claims.
- [ ] Readme should present one-layer acceptance path: 3 core claims first, then optional extensions.

## MVP-only required checks
- [ ] Keep baseline acceptance on `advisor`/`mvp` target outputs only.
- [ ] If additional items appear as `full` target outputs, treat them as roadmap evidence, not MVP acceptance.




