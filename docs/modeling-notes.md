# Modeling Notes

## Current Validation Setup

Validation masks a contiguous interval inside each training well and predicts the hidden interval from inference-available inputs. This is closer to the Kaggle rerun setup than random row splits because adjacent rows along a lateral well are strongly autocorrelated.

Current default mask:

- Start fraction: `0.65`
- Zone fraction: `0.20`

## Baseline Results

Full-train interpolation baseline:

- Wells: 773
- Masked rows: 1,018,140
- Weighted RMSE: 122.3599
- Median per-well RMSE: 44.2213
- Worst per-well RMSE: 1517.3749

First 50 wells, interpolation only:

- Weighted RMSE: 84.9903
- Median per-well RMSE: 49.1814

First 50 wells, interpolation plus typewell gamma-ray correlation blend:

- Weighted RMSE: 68.0063
- Median per-well RMSE: 50.6428

Interpretation: the first GR-correlation blend reduces large weighted errors on the sampled wells, but it slightly worsens the median well. That means it should remain an experimental component, not a default replacement for interpolation.

## Important Data Constraint

The public test horizontal files do not include train-only formation-surface columns such as `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, and `BUDA`. Public test typewell files also omit the train `Geology` label. A valid Kaggle notebook should prioritize features available at inference:

- Horizontal `MD`, `X`, `Y`, `Z`, `GR`, and `TVT_input`
- Typewell `TVT` and `GR`
- Features derived from those inputs

Train-only columns may still help exploratory diagnosis, but relying on them for test inference would fail or leak.

## Next Modeling Direction

1. Tune the GR-correlation blend by well class rather than applying one global weight.
2. Add confidence features for correlation quality, then use correlation only when it beats continuity locally.
3. Train a residual model that predicts correction to interpolation from trajectory and gamma-ray features.
4. Use whole-well holdout validation in addition to masked-zone validation.
5. Convert the final pipeline into a Kaggle notebook that can run with internet disabled.
