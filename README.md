# ROGII Wellbore Geology Prediction

Domain-informed machine learning workflow for the Kaggle competition
[ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).

The task is to predict `TVT` (True Vertical Thickness) for hidden evaluation zones along horizontal wells using trajectory, gamma ray, geological surface, and typewell reference data.

## Goals

- Build a valid first Kaggle notebook submission.
- Establish leakage-aware validation by holding out wells and masking contiguous evaluation zones.
- Develop petroleum-engineering-informed features around well trajectory, gamma ray correlation, formation surfaces, and typewell references.
- Keep the repository reproducible without redistributing competition data.

## Data Policy

Competition data is subject to Kaggle competition rules and must not be redistributed here.

Expected local layout after accepting the rules and downloading data from Kaggle:

```text
data/
  raw/
    train/
    test/
    sample_submission.csv
    AI_wellbore_geology_prediction_task_en.pptx
```

The `data/`, `models/`, `submissions/`, and large generated artifacts are ignored by git.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

After placing Kaggle data under `data/raw`, create a baseline submission:

```bash
python scripts/make_baseline_submission.py --data-dir data/raw --output submissions/submission.csv
```

## Current Baseline

The first baseline is deliberately conservative:

1. Read each test horizontal well file.
2. Identify rows requiring prediction from `sample_submission.csv`.
3. Use `TVT_input` where available.
4. Interpolate and extrapolate missing values from nearby known `TVT_input`.
5. Fall back to a smooth `Z`-based estimate if a well has no usable `TVT_input`.

This gives a valid submission path quickly while leaving room for stronger ML and typewell-correlation models.

## Planned Work

- EDA notebook with well-level summaries and visual checks.
- Validation harness that masks contiguous zones inside training wells.
- Feature pipeline for trajectory, gamma ray, formation-surface distances, and local trend features.
- LightGBM/CatBoost tabular baseline.
- Typewell gamma-ray correlation features.
- Portfolio write-up focused on honest methodology and lessons learned.

## Non-Goals

- No raw competition data in git.
- No claims about production geology automation from a single Kaggle result.
- No leaderboard claims without reproducible validation and actual submitted scores.
