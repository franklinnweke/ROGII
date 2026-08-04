# ROGII Wellbore Geology Prediction - conservative saturated-ramp baseline
#
# Kaggle submission notebook draft.
# This version is intentionally simple, reproducible, and submission-safe:
# it discovers the hidden rerun wells dynamically, uses only inference-available
# columns, preserves sample order, and writes /kaggle/working/submission.csv.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("data/raw"),
        Path("../data/raw"),
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


def saturated_ramp_prediction(horizontal_df: pd.DataFrame) -> pd.Series:
    """Blend flat continuity with a damped trend fitted to the last 80 known rows."""
    tvt_input = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce")
    flat = tvt_input.ffill().bfill().astype(float)
    md = pd.to_numeric(horizontal_df["MD"], errors="coerce")
    known = np.flatnonzero(tvt_input.notna().to_numpy())
    if len(known) >= 2:
        last = int(known[-1])
        tail = known[-80:]
        tail = tail[md.iloc[tail].notna().to_numpy()]
        hidden = np.arange(last + 1, len(horizontal_df))
        if len(tail) >= 2 and len(hidden) and np.ptp(md.iloc[tail].to_numpy()) > 0:
            slope = float(
                np.polyfit(
                    md.iloc[tail].to_numpy(dtype=float),
                    tvt_input.iloc[tail].to_numpy(dtype=float),
                    1,
                )[0]
            )
            distance = md.iloc[hidden].to_numpy(dtype=float) - float(md.iloc[last])
            if np.isfinite(distance).all():
                distance = np.maximum(distance, 0.0)
                ramp = float(tvt_input.iloc[last]) + slope * 100.0 * (
                    1.0 - np.exp(-distance / 100.0)
                )
                weight = 1.0 / (
                    1.0 + np.exp(-np.clip((distance - 100.0) / 100.0, -60, 60))
                )
                flat.iloc[hidden] = float(tvt_input.iloc[last]) + weight * (
                    ramp - float(tvt_input.iloc[last])
                )
                return flat.rename("tvt")

    if tvt_input.notna().any():
        return flat.rename("tvt")

    # Defensive fallback; competition wells are expected to have a known prefix.
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
        tvt = saturated_ramp_prediction(horizontal)
        for row_index in row_indices:
            predictions[f"{well_id}_{row_index}"] = float(tvt.iloc[row_index])

    submission = sample[["id"]].copy()
    submission["tvt"] = submission["id"].map(predictions)
    if submission["tvt"].isna().any():
        missing = submission.loc[submission["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Missing predictions for ids: {missing}")
    if list(submission.columns) != ["id", "tvt"] or not np.isfinite(submission["tvt"]).all():
        raise ValueError("Generated submission failed schema or finite-value validation.")
    return submission


def main() -> None:
    root = find_competition_root()
    output_dir = (
        Path("/kaggle/working")
        if Path("/kaggle/working").exists()
        else root.resolve().parent.parent / "submissions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = make_submission(root)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Competition root: {root}")
    print(f"Wrote {output_path} with {len(submission)} rows")
    print(submission.head())


if __name__ == "__main__":
    main()
