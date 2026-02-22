# Appendix G - GKP Toy Proxy

## Scope and role

This appendix is non-MVP and for intuition only.  
`GKP proxy` is **appendix-only** and is excluded from the README acceptance boundary.

## Scope

This appendix contains a minimal, synthetic GKP-oriented sanity indicator:

- `scripts/run_gkp_sweep.py` computes a toy logical-error proxy from finite squeezing `r` and additive shift noise `sigma`.
- The resulting plot (`gkp_proxy`) is shown as a **supporting intuition example**, not a core bus control claim.

## What this appendix is (and is not)

- **In scope**: one-page qualitative mapping of squeezing vs shift-noise to a logical error proxy.
- **Not in scope**: full GKP encoding/decoding stack, syndrome extraction, or full fault-tolerance modeling.

## Reproduce

```bash
python scripts/run_gkp_sweep.py --profile both
```

This writes:

- `assets/gkp_proxy.png`
- `assets/web/gkp_proxy.png`
- `assets/paper/gkp_proxy.png` (for `--profile both`)

## Review note

The appendix is intentionally separated from the main scenario gallery so the reviewer focus stays on MVP proof points:

1. calibrated squeezing workflow (intrinsic vs observed),
2. propagation loss behavior,
3. control-constraint sensitivity (one sweep).


