# Baseline Audit Report (Phase 0)

## Scope and authority

- Archive note: this file is historical documentation only.  
- Non-blocking for MVP acceptance; roadmap-only artifacts in this report should be treated as exploratory evidence.
- This is a historical point-in-time report and is not a live defect list for current docs.

Date: 2026-02-18

Scope:
- No code behavior changes made.
- Read-only audit of documentation formatting, plotting/script label sources, and current asset dimensions/sizes.

## 1) README.md formatting issues (current state)

- Encoding artifacts in section headings and text (mojibake) indicate non-UTF-8 rendering and broken formatting:
  - `## Model Definitions and Assumptions` (line 221).
  - `## Testing & CI` (line 305).
  - `## Project Structure` (line 326).
  - Multiple non-ASCII symbols appear as mojibake (examples around lines 17, 97, 103, 109-111, 229-230, 255-259, 299-301, 329-347).
- Heading/title readability is effectively broken by those encoded glyphs, which also affects generated anchors and section discoverability.
- Mixed-language / corrupted symbol blocks:
  - The "Project Structure" box-drawing block appears garbled (lines 329-347) instead of plain ASCII/Unicode tree characters.
- Minor line-flow consistency issues:
  - Several long heading/annotation lines include mixed punctuation/Unicode that can render unevenly across viewers, notably in "Phase 1 / Phase 2 / Phase 3" lines 97-107 and model-notes line 223 onward.

## 2) Plot/layout label sources in generator scripts

- `scripts/generate_calibration_demo.py`
  - Figure is built in `configure_axes()` and rendered by `run_animation()`:
    - 2x2 grid with panel split (`height_ratios`, `width_ratios`).
    - Global style from `apply_ieee_style(base_font_size=10, tick_font_size=9, dpi=...)` (`run_animation`).
  - Dashboard panel labels and values:
    - `draw_dashboard()`: phase title, pump, loss bars, r, intrinsic/observed squeezing, transmissivity.
  - Wigner panel labels:
    - `draw_wigner_panel()`: x label `x (position quadrature)`, y label `p (momentum quadrature)`.
  - Calibration-axis labels:
    - `draw_calibration_panel()`: `Pump power P (mW)`, `Squeezing (dB)`.
  - Output:
    - `calibration_demo.gif` (rendered via PIL optimization at command-configured `gif_colors`).

- `scripts/generate_dashboard_gallery.py`
  - Scenario renderers (`scenario_vacuum`, `scenario_calibration`, `scenario_decoherence`) set all axis text via:
    - `_draw_calibration_curve()` -> `style_axis(..., xlabel="Pump power (mW)", ylabel="Intrinsic squeezing (dB)")`
    - `_style_wigner_panel()` -> `xlabel="x (SNU)", `ylabel="p (SNU)"`
    - `bar`/`text` labels in variance panel include `Variance (SNU; vacuum=0.5)`
  - Figure text and suptitles are set per scenario.
  - Output PNGs saved at `dpi=300` through `save_ieee(...)`:
    - `dashboard_vacuum.png`
    - `dashboard_calibration.png`
    - `dashboard_decoherence.png`

- `scripts/generate_advanced_dashboard_gallery.py`
  - `scenario_multimode()`, `scenario_topology()`, `scenario_digital_twin()` define panel labels via `style_axis` and inline `set_xlabel`/`set_ylabel`.
  - Typical labels include:
    - `Time-bin index (unitless)`
    - `Observed squeezing (dB)`
    - `Observed anti-sq (dB)`
    - `Squeezing (dB)` etc.
  - Output PNGs saved at `dpi=300` via `save_ieee(...)`:
    - `dashboard_multimode.png`
    - `dashboard_topology.png`
    - `dashboard_digital_twin.png`

- `scripts/generate_*_gif.py` (present in repo)
  - `generate_scenario_gallery_gif.py`
    - Reads `dashboard_*.png` and adds corner labels (`Scenario 1..3`) with `load_labeled_image()` and `ImageDraw`.
    - Label font is `DejaVuSans` if available, fallback to PIL default.
    - Frames are crossfaded then quantized.
  - `generate_advanced_gallery_gif.py`
    - Same flow for `Advanced 1..3`, overlay font/path as above.
  - `generate_advanced_evidence_gif.py`
    - Composes:
      - two-panel sweep slide from `sweep_latency.png` + `sweep_quantization.png`
      - single slides from `gkp_proxy.png` and `drift_recovery.png`
    - Labels and titles are manually overlaid (`Evidence 1..3`, `Fault tolerance / GKP proxy`, etc.) with DejaVuSans fallback.

## 3) Current asset inventory (width/height/file size)

| Asset | Width (px) | Height (px) | File size (bytes) |
|---|---:|---:|---:|
| calibration_demo.gif | 984 | 510 | 2,332,667 |
| dashboard_vacuum.png | 1,939 | 1,369 | 242,725 |
| dashboard_calibration.png | 2,032 | 1,369 | 243,959 |
| dashboard_decoherence.png | 2,468 | 1,472 | 308,008 |
| dashboard_topology.png | 2,127 | 1,167 | 140,311 |
| dashboard_multimode.png | 2,035 | 1,167 | 245,215 |
| dashboard_digital_twin.png | 2,253 | 1,167 | 298,091 |
| sweep_latency.png | 1,080 | 630 | 42,660 |
| sweep_quantization.png | 1,080 | 630 | 38,706 |
| gkp_proxy.png | 1,650 | 675 | 94,479 |
| drift_recovery.png | 1,500 | 1,350 | 237,005 |

## 4) Recommended target specs for cleanup

- Formatting/rendering
  - Normalize `README.md` to UTF-8 without BOM and replace mojibake with valid Unicode characters.
  - Regenerate section headings and symbols consistently (especially lines near 221, 305, 326) and verify all symbols are cleanly encoded.
  - Recheck mermaid/tree blocks for valid UTF-8 rendering.
- Plot layout / clipping
  - Centralize typography and axis styling in one place (single font family and sizes for titles/labels/ticks across all generators).
  - Keep panel spacing and subtitle margins consistent; enforce `bbox_inches="tight"`/`pad_inches`/`constrained_layout` or equivalent so x-axis labels are not clipped.
  - Standardize canvas padding before GIF composition; avoid label overlays too close to borders.
- Units and annotations
  - Ensure all physical axes include explicit units in labels (mW, dB, SNU, etc.) and avoid ad-hoc unit text.
- Export targets (IEEE-like baseline)
  - PNG: 300 DPI, stable target width policy, and consistent typography.
  - GIF: fixed source canvas size per family and deterministic frame cadence.
