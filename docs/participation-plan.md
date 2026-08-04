# Participation Plan

## First Submission Path

1. Accept the competition rules on Kaggle.
2. Download the competition data locally or attach it to a Kaggle Notebook.
3. Run the offline saturated-ramp notebook and submit it through Kaggle Code.
4. Record public leaderboard score, private notes, runtime, notebook version, and artifact checksum.

Current result: version 1 succeeded with public score `15.660`. The local method scored `15.5281` weighted RMSE on all 773 training wells under original-gap validation.

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

1. Flat-tail and saturated-ramp baselines from `TVT_input`.
2. Typewell-correlation features from local `GR` pattern matching, retained only if they beat the baseline under honest validation.
3. Spatial/regime-aware features using trajectory, azimuth/dip proxies, neighboring wells, and recent TVT trend.
4. Guarded selection between continuity and learned corrections.
5. Portfolio write-up and post-competition reproducibility package.

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
