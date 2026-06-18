# Modeling Notes

## Current Validation Setup

Validation should use the original `TVT_input` gaps in each training well. Kaggle's public examples and public notebooks confirm these gaps are contiguous tails, and `sample_submission.csv` row IDs map directly to those hidden rows.

Synthetic masks remain useful for robustness checks, but they are secondary.

## Baseline Correction

The first implementation used interpolation/extrapolation through the hidden zone. Public Kaggle analysis shows the stronger zero-order baseline is simpler: carry the last known `TVT_input` value through the hidden tail.

That flat-tail baseline is now the default for submission generation.

## Previous Baseline Results

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

Interpretation: the first GR-correlation blend reduced large weighted errors under the old synthetic-mask validation, but that validation target was not aligned with the real eval-tail task. It should be retested against original `TVT_input` gaps before being trusted.

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
