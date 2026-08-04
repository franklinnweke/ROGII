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

For the deadline-driven agent and Antigravity execution contract, start with
[`docs/agent-execution-handoff.md`](docs/agent-execution-handoff.md).

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

For the notebook-only competition submission path, build and test the offline notebook:

```bash
python scripts/build_kaggle_notebook.py
jupyter nbconvert --to notebook --execute notebooks/rogii_submission.ipynb \
  --output rogii_submission.executed.ipynb
```

Kaggle replaces the three public example test wells with roughly 200 hidden wells during the
notebook rerun. The notebook discovers those wells from the hidden `sample_submission.csv` and
writes the required `/kaggle/working/submission.csv`; the local 14,151-row CSV is not uploaded
directly in this code competition.

## Current Baseline

The current submitted baseline is deliberately conservative:

1. Read each test horizontal well file.
2. Identify rows requiring prediction from `sample_submission.csv`.
3. Use `TVT_input` where available.
4. Fit a recent trend to the last known `TVT_input` rows.
5. Apply a damped, saturated ramp from the last known value, preserving continuity.
6. Fall back to flat continuity or a smooth `Z`-based estimate when the required inputs are unavailable.

The method was validated on all 773 training wells using the original contiguous `TVT_input` gaps:

- Saturated-ramp weighted RMSE: `15.5281`
- Flat-tail weighted RMSE: `15.9099`
- Kaggle public score: `15.660`
- Kaggle submission status: succeeded

This is a valid first competition result, not a production geology or geosteering claim. The public score is substantially behind the leading systems, so further modeling should be treated as an experiment rather than a guaranteed improvement.

## Planned Work

- Preserve the submitted notebook, checksum, validation report, and execution record in GitHub.
- Repair and rerun the complete test suite in a stable virtual environment.
- Evaluate one bounded spatial/regime-aware improvement using inference-available data.
- Publish a technical write-up covering the real score, failed experiments, domain assumptions, and limitations.

## Non-Goals

- No raw competition data in git.
- No claims about production geology automation from a single Kaggle result.
- No leaderboard claims without reproducible validation and actual submitted scores.
