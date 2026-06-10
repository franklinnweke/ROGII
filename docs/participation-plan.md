# Participation Plan

## First Submission Path

1. Accept the competition rules on Kaggle.
2. Download the competition data locally or attach it to a Kaggle Notebook.
3. Run the interpolation baseline and submit `submission.csv`.
4. Record public leaderboard score, private notes, runtime, and notebook version.

## EDA Checklist

- Count wells, row counts, and hidden-zone lengths.
- Inspect missingness by column and by well.
- Plot `MD`, `Z`, `TVT_input`, `GR`, and formation surfaces for representative wells.
- Compare horizontal `GR` signatures against typewell `GR` signatures.
- Identify wells where formation surfaces conflict with observed trajectory or log behavior.
- Segment wells by lateral length, structural dip, and log noise.

## Validation Strategy

Use validation that respects geology and hidden-test mechanics:

- Hold out entire wells for generalization checks.
- Mask contiguous intervals inside training wells to simulate hidden evaluation zones.
- Report RMSE by well, zone length, and apparent geology interval.
- Avoid random row splits because neighboring rows are highly autocorrelated.

## Modeling Roadmap

1. Interpolation/extrapolation baseline from `TVT_input`.
2. Tabular model with trajectory, gamma ray, rolling-window, and formation-surface-distance features.
3. Typewell-correlation features from local `GR` pattern matching.
4. Ensemble of smooth baseline plus ML residual model.
5. Post-processing for continuity and physically plausible local trends.

## Domain Assumptions To Test

- `GR` signatures can help correlate horizontal intervals to vertical typewell TVT positions.
- Formation surfaces provide useful structural anchors, but may be noisy or insufficient alone.
- Smooth TVT evolution is a reasonable prior along short lateral intervals.
- Abrupt changes should require evidence from trajectory/log/surface context.

## Portfolio Deliverables

- Kaggle notebook with EDA, validation, first baseline, and score.
- GitHub repository with code, tests, and clear data-use boundaries.
- Technical write-up focused on methodology and domain reasoning.
- LinkedIn post describing participation, not overstated outcomes.
- Portfolio case study once there is a real submitted score and meaningful iteration history.
