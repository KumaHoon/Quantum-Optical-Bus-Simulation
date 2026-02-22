# Figure Verification Functions (Contract Driven)

This document is the authoritative reference for the figure verification engine implemented by
`scripts/figure_verification.py` and used by `scripts/verify_assets_profiles.py`.

All checks are driven by `docs/FIGURE_CONTRACT.yaml` and operate on:
- generated image artifacts (`PNG/PDF/GIF`)
- companion numeric data (`assets/<profile>/data/<figure_id>.npz`)
- companion metadata (`assets/<profile>/meta/<figure_id>.meta.json`)

## Error format (canonical)

Every failure is emitted with stable code and diagnostic context:

`VAL<NNN> <figure_id>/<profile> <validator>: <short reason> | <diagnostics>`

Example:
`VAL130 dashboard_decoherence/web loss_monotonic_observed: distance-to-zero increased at idx=7 | loss_db=6.0->7.0, d=2.1->2.5`

## Generic checks

### VAL001 `outputs_exist`
Required image outputs (web/paper) declared in `docs/FIGURE_CONTRACT.yaml` must exist for each target figure.

### VAL002 `companions_exist`
Required companion files must exist:
- `assets/<profile>/meta/<figure_id>.meta.json`
- `assets/<profile>/data/<figure_id>.npz` where required by contract

### VAL003 `meta_schema_valid`
Required metadata fields:
- `figure_id`, `profile`, `created_at_utc`
- `generator_script`, `labels`, `units`, `canvas_px`, `dpi`, `notes`

### VAL004 `canvas_and_dpi_match_profile`
`canvas_px` and `dpi` must match profile contract:
- web: exact pixel size and DPI
- paper: exact size with small tolerance for floating-point conversions

### VAL005 `units_required_present`
Each required label/unit string in contract must appear in either:
- `meta.labels.title/xlabel/ylabel/y2label`
- `meta.units` values

## Validator handlers

All function validators are listed below.

## VAL101 `no_nan_inf`

Input: `keys: [ ... ]`

For each requested key:
- fail with `VAL101` if key is missing (alias-resolution is attempted for known legacy names)
- fail with `VAL102` if any non-finite values appear (NaN/Inf)

## VAL110 `heisenberg_sanity`

Input: `var_x_key`, `var_p_key`, `vacuum_var=0.5` (default)

Check:
- `min(var_x * var_p) >= vacuum_var^2 - tol`

## VAL120 `probability_distribution`

Input: `p_key`, `tol_sum_abs=1e-3`, `require_nonnegative=true`

Checks:
- optional non-negativity
- normalization: `abs(sum(P)-1) <= tol_sum_abs`

## VAL121
Same contract branch as `probability_distribution` when normalization exceeds tolerance.

## VAL130 `loss_monotonic_observed`

Input: `loss_key`, `observed_sq_key`, optional `direction` (`toward_zero`)

For increasing loss, observed squeezing distance to zero must not increase by more than `tol_abs`.

## VAL140 `intrinsic_ge_observed`

Input: `intrinsic_sq_key`, `observed_sq_key`

Require:
- `abs(intrinsic) >= abs(observed) - tol_abs`

## VAL150 `integer_axis`

Input: `axis`

Axis values must be integer-valued within tolerance `tol`.

## VAL160 `matrix_symmetric`

Input: `key`

Matrix must be square and symmetric within `tol`.

## VAL161
Same branch for asymmetry beyond tolerance.

## VAL170 `corr_diagonal_one`

Input: `key`, `tol`

Correlation diagonal must be all ones within tolerance.

## VAL180 `fit_reasonable_range`

Input: `eta_key`, `eta_min`, `eta_max`

Values must remain within bounds.

## VAL190 `trend_degrades`

Input: `x_key`, `y_key`, `expectation`, `tol`, `max_violations`

For synthetic sweeps, enforce non-improving trend with `x`-sorted ordering.

## VAL200 `recovery_target`

Input: `time_key`, `metric_key`, `target_time_h`, `target_fraction_recovered`, `steady_state_max_db`

Enforces drift recovery fraction and steady-state bound.

## VAL201
Same handler branch for insufficient recovery/steady-state.

## VAL210/VAL211 `gif_multiframe`

Input: `web_gif`, `min_frames`

- `VAL210`: GIF file missing
- `VAL211`: fewer than `min_frames` animation frames

## Verification output

`python scripts/verify_assets_profiles.py --profile web --target advisor --verify`
must fail fast on first hard validation failure when `--strict` is used.
The verifier is intentionally not based on image hashing; it validates physics and metadata through
`.meta.json` and `.npz` content.
