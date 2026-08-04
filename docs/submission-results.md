# Kaggle Submission Results

## Baseline checkpoint

- Competition: [ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- Submission status: Succeeded
- Public score: `15.660`
- Local original-gap validation: `15.5281` weighted RMSE
- Submission reference: `55248683`
- Kaggle script version: `340214222`
- Notebook: `notebooks/rogii_submission.ipynb`

## Method

The submitted baseline uses `TVT_input` where available, fits a short recent trend over the last known segment, and applies a bounded saturated ramp into the hidden tail. It falls back to a flat continuation when the trend is unusable. The design is intentionally conservative because validation showed unconstrained linear extrapolation was unstable.

## Interpretation

This is a successful, reproducible first submission and a useful portfolio checkpoint. It is not a competitive leaderboard result: the public rank was 5009 at review time. The next experiment should focus on guarded gamma-ray/typewell correlation and spatial or regime-aware features, with every change compared against this baseline.

## Reproducibility

Raw Kaggle data and generated submission files are excluded from Git. The notebook, source implementation, validation utilities, and documentation are public; the local submission checksum and full report remain available in the working environment.
