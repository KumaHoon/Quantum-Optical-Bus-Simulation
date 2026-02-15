# Data Schema (Digital Twin Calibration)

This project supports CSV-based calibration data for fitting simulator parameters.

## Required columns
- `timestamp`
  - Numeric step index or epoch-like timestamp.
- `pump_power_mw`
  - Pump power in mW.

And at least one measurement set:
- Variance form:
  - `measured_var_x`
  - optionally `measured_var_p`
- Or squeezing form:
  - `measured_squeezing_db`

## Optional columns
- `estimated_loss_db`
  - Initial loss estimate (used as optimizer initial guess).
- `phase_estimate`
  - Estimated phase at each timestamp.

## Notes
- All numeric columns are parsed as `float`.
- Empty cells are ignored.
- Current estimator (`fit_eta_and_loss`) assumes global constants for:
  - `eta` in `r = eta * sqrt(P)`
  - `loss_db` for transmissivity conversion `T = 10^(-loss_db/10)`

## Minimal CSV example
```csv
timestamp,pump_power_mw,measured_var_x,measured_var_p,estimated_loss_db,phase_estimate
0,10,0.470,0.531,1.0,0.01
1,25,0.426,0.588,1.0,0.03
2,50,0.364,0.693,1.0,0.05
```
