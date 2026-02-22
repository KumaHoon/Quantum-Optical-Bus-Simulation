# Units and Conventions (Single Source of Truth)

This document defines variable names, physical meanings, units, and reporting conventions used across code, plots, and verification.

## Core physical conventions

### Quadrature normalization

- All variance-like quantities are reported in convention: vacuum variance = `0.5`.
- Internal representations must be converted to this convention before reporting and plotting.

### Loss model

- Loss is represented as `loss_db` in dB.
- `loss_db >= 0` where `0 dB` means no loss.
- Transmissivity:
  - `T = 10^(-loss_db / 10)`
- Conceptual pure-loss channel model:
  - `a_out = sqrt(T) * a_in + sqrt(1-T) * a_vac`

### Squeezing in dB

Let `V` be quadrature variance in vacuum=0.5 convention.

- `S_dB = 10 * log10(V / 0.5)`
- `S_dB < 0`: squeezed (below vacuum)
- `S_dB = 0`: vacuum level
- `S_dB > 0`: anti-squeezed

### Intrinsic vs observed

- `intrinsic`: before loss.
- `observed`: after loss.
- As loss increases, observed squeezing should move toward `0 dB` and should not be more extreme than intrinsic.

## Canonical variable dictionary

### Pump power

- Name: `pump_power_mw`
- Unit: `mW`
- Label: `Pump power [mW]`

### Proxy coefficient

- Name: `eta`
- Unit: `1/sqrt(mW)`
- Label: `eta [1/sqrt(mW)]`
- Usage: phenomenological placeholder unless fitted from calibration data

### Squeezing parameter

- Name: `r`
- Unit: dimensionless
- Label: `Squeezing parameter r [-]`

### Phase / rotation

- Name: `phase_rad`
- Unit: `rad`
- Label: `Phase [rad]`

### Variances

- Names: `var_x`, `var_p`
- Unit: dimensionless
- Labels:
  - `Var(x) [vac=0.5]`
  - `Var(p) [vac=0.5]`

### Squeezing in dB

- Names: `sq_db_x`, `sq_db_p` (or `squeezing_db_x`, etc.)
- Unit: `dB`
- Labels:
  - `Squeezing X [dB]`
  - `Squeezing P [dB]`

### Photon number distribution

- Name: `p_n`
- Unit: probability
- Labels:
  - X-axis: `Photon number n [-]`
  - Y-axis: `Probability P(n) [-]`
- Constraints:
  - `p_n >= 0`
  - `sum(p_n) ~= 1` (tolerance defined in verification)

### Topology and time-bin indices

- Names: `mode_index`, `time_bin_index`
- Unit: index (dimensionless)
- Labels:
  - `Mode index [-]`
  - `Time-bin index [-]`

### Correlations and covariance

- Names: `corr_x`, `corr_p`, `cov_x`, `cov_p`
- Unit: dimensionless
- Labels:
  - `Corr(X) [-]`, `Corr(P) [-]`
  - `Cov(X) [vac=0.5]`, `Cov(P) [vac=0.5]` when variance normalized
- Required properties:
  - Correlation matrices are symmetric.
  - Diagonal of correlation matrices is one.

### Latency and quantization

- `latency_bins` : integer, unit `bins`
- `latency_cycles` : integer, unit `cycles`
- `quantization_step` : integer
- Common index-labels:
  - `Latency [bins]`
  - `Latency [cycles]`
  - `Quantization [-]`

### Fixed-point HDL interface (context)

- Q-format: `Q1.15`
- Signed two's-complement.
- Metadata should include:
  - qformat string
  - rounding mode
  - saturation policy

## Figure companion expectations

### Numeric companion (`.npz`)

Each figure that reports numeric metrics must provide a companion:
- `assets/<profile>/data/<figure_id>.npz`
- Arrays must use canonical names above where possible.

### Metadata companion (`.meta.json`)

Each figure must provide:
- `assets/<profile>/meta/<figure_id>.meta.json`
- `units` dictionary and plot labels used in this figure.

## Verification invariants

The following invariants are expected by contract checks:

- Probability: `P(n) >= 0` and `sum(P(n)) ~= 1`
- Heisenberg sanity: `var_x * var_p >= (0.5)^2`
- Loss monotonicity: observed squeezing distance to 0 dB should not increase with larger loss
- Trend checks: sweep metrics should not improve as latency/quantization constraints worsen
- Matrix checks: symmetry and diagonal-one requirements where applicable
