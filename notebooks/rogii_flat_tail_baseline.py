# ROGII Wellbore Geology Prediction - flat-tail baseline
#
# Kaggle submission notebook draft.
# This version is intentionally simple, reproducible, and submission-safe:
# it uses only columns available in the test rerun and writes submission.csv.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("data/raw"),
    ]
    for candidate in candidates:
        if (candidate / "sample_submission.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find ROGII competition input directory.")


def parse_submission_ids(sample_submission: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for value in sample_submission["id"].astype(str):
        well_id, row_index = value.rsplit("_", 1)
        groups.setdefault(well_id, []).append(int(row_index))
    return groups


def flat_tail_prediction(horizontal_df: pd.DataFrame) -> pd.Series:
    """Carry the last known TVT_input value through the hidden evaluation tail."""
    tvt_input = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce")
    if tvt_input.notna().any():
        return tvt_input.ffill().bfill().rename("tvt")

    # Defensive fallback. The competition data should have a known prefix.
    if "Z" in horizontal_df.columns:
        return pd.to_numeric(horizontal_df["Z"], errors="coerce").interpolate(
            limit_direction="both"
        ).rename("tvt")
    return pd.Series(np.zeros(len(horizontal_df), dtype=float), index=horizontal_df.index, name="tvt")


def make_submission(root: Path) -> pd.DataFrame:
    sample = pd.read_csv(root / "sample_submission.csv")
    requested_rows = parse_submission_ids(sample)
    predictions: dict[str, float] = {}

    for well_id, row_indices in requested_rows.items():
        horizontal_path = root / "test" / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path)
        tvt = flat_tail_prediction(horizontal)
        for row_index in row_indices:
            predictions[f"{well_id}_{row_index}"] = float(tvt.iloc[row_index])

    submission = sample[["id"]].copy()
    submission["tvt"] = submission["id"].map(predictions)
    if submission["tvt"].isna().any():
        missing = submission.loc[submission["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Missing predictions for ids: {missing}")
    return submission


def main() -> None:
    root = find_competition_root()
    output_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("submissions")
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = make_submission(root)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Competition root: {root}")
    print(f"Wrote {output_path} with {len(submission)} rows")
    print(submission.head())


if __name__ == "__main__":
    main()
