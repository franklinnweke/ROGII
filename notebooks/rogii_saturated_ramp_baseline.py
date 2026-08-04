# ROGII Wellbore Geology Prediction - conservative saturated-ramp baseline
#
# Kaggle submission notebook source (built into notebooks/rogii_submission.ipynb).
# This version is intentionally simple, reproducible, and submission-safe:
# it discovers the hidden rerun wells dynamically, uses only inference-available
# columns, preserves sample order, and writes /kaggle/working/submission.csv.
#
# Filename note: previously notebooks/rogii_flat_tail_baseline.py; method is
# saturated-ramp with flat continuity as the guarded fallback.

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Keep in sync with rogii_geology.baseline defaults used by local package path.
MIN_MD_SPAN_FT = 1.0
MAX_ABS_SLOPE = 5.0
RAMP_TAU = 100.0
RAMP_CENTER = 100.0
RAMP_SCALE = 100.0
RAMP_GAIN = 1.0
RAMP_TAIL_ROWS = 80


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


def saturated_ramp_prediction(
    horizontal_df: pd.DataFrame,
    meta: dict | None = None,
) -> pd.Series:
    """Blend flat continuity with a damped trend fitted to the last known rows."""
    tvt_input = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce")
    flat = tvt_input.ffill().bfill().astype(float)
    md = pd.to_numeric(horizontal_df["MD"], errors="coerce")
    known = np.flatnonzero(tvt_input.notna().to_numpy())

    def _fallback(reason: str, series: pd.Series) -> pd.Series:
        if meta is not None:
            meta["used_ramp"] = False
            meta["fallback_reason"] = reason
        return series.rename("tvt")

    if len(known) >= 2:
        last = int(known[-1])
        tail = known[-RAMP_TAIL_ROWS:]
        tail = tail[md.iloc[tail].notna().to_numpy()]
        hidden = np.arange(last + 1, len(horizontal_df))
        if len(tail) >= 2 and len(hidden):
            tail_md = md.iloc[tail].to_numpy(dtype=float)
            md_span = float(np.ptp(tail_md))
            if np.isfinite(md_span) and md_span >= MIN_MD_SPAN_FT:
                slope = float(
                    np.polyfit(
                        tail_md,
                        tvt_input.iloc[tail].to_numpy(dtype=float),
                        1,
                    )[0]
                )
                if np.isfinite(slope) and abs(slope) <= MAX_ABS_SLOPE:
                    distance = md.iloc[hidden].to_numpy(dtype=float) - float(md.iloc[last])
                    if np.isfinite(distance).all():
                        distance = np.maximum(distance, 0.0)
                        last_tvt = float(tvt_input.iloc[last])
                        ramp = last_tvt + slope * RAMP_TAU * (
                            1.0 - np.exp(-distance / RAMP_TAU)
                        )
                        weight = 1.0 / (
                            1.0
                            + np.exp(
                                -np.clip(
                                    (distance - RAMP_CENTER) / RAMP_SCALE, -60, 60
                                )
                            )
                        )
                        flat.iloc[hidden] = last_tvt + RAMP_GAIN * weight * (
                            ramp - last_tvt
                        )
                        if np.isfinite(flat.to_numpy(dtype=float)).all():
                            if meta is not None:
                                meta["used_ramp"] = True
                                meta["fallback_reason"] = None
                                meta["slope"] = slope
                                meta["md_span"] = md_span
                            return flat.rename("tvt")
                        return _fallback("non_finite_ramp_output", flat)
                    return _fallback("non_finite_md_distance", flat)
                return _fallback("slope_unstable_or_extreme", flat)
            return _fallback("md_span_too_small", flat)
        if len(tail) < 2:
            return _fallback("insufficient_tail_with_md", flat)
        return _fallback("no_hidden_tail", flat)

    if tvt_input.notna().any():
        return _fallback("insufficient_known_tvt", flat)

    # Defensive fallback; competition wells are expected to have a known prefix.
    # Prefer failing loudly in the run summary rather than inventing TVT from Z.
    if meta is not None:
        meta["used_ramp"] = False
        meta["fallback_reason"] = "no_usable_tvt_input"
    if "Z" in horizontal_df.columns:
        z_series = pd.to_numeric(horizontal_df["Z"], errors="coerce").interpolate(
            limit_direction="both"
        )
        if z_series.notna().any():
            return z_series.rename("tvt")
    return pd.Series(
        np.zeros(len(horizontal_df), dtype=float),
        index=horizontal_df.index,
        name="tvt",
    )


def make_submission(root: Path) -> pd.DataFrame:
    sample = pd.read_csv(root / "sample_submission.csv")
    requested_rows = parse_submission_ids(sample)
    predictions: dict[str, float] = {}
    ramp_applied = 0
    fallback_reasons: Counter[str] = Counter()

    for well_id, row_indices in requested_rows.items():
        horizontal_path = root / "test" / f"{well_id}__horizontal_well.csv"
        horizontal = pd.read_csv(horizontal_path)
        n_rows = len(horizontal)
        meta: dict = {}
        tvt = saturated_ramp_prediction(horizontal, meta=meta)
        if meta.get("used_ramp"):
            ramp_applied += 1
        else:
            fallback_reasons[str(meta.get("fallback_reason") or "unknown")] += 1

        for row_index in row_indices:
            if not 0 <= row_index < n_rows:
                raise ValueError(
                    f"Submission id {well_id}_{row_index} is outside test-well "
                    f"row bounds 0..{n_rows - 1}"
                )
            predictions[f"{well_id}_{row_index}"] = float(tvt.iloc[row_index])

    submission = sample[["id"]].copy()
    submission["tvt"] = submission["id"].map(predictions)
    if submission["tvt"].isna().any():
        missing = submission.loc[submission["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Missing predictions for ids: {missing}")
    if list(submission.columns) != ["id", "tvt"] or not np.isfinite(submission["tvt"]).all():
        raise ValueError("Generated submission failed schema or finite-value validation.")

    print(
        f"wells={len(requested_rows)} ramp_applied={ramp_applied} "
        f"fallback_reasons={dict(fallback_reasons)}"
    )
    if ramp_applied == 0 and len(requested_rows) > 0:
        print(
            "WARNING: saturated ramp applied to zero wells; "
            "submission is effectively flat/Z fallback."
        )
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
