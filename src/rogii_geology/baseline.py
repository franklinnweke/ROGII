from __future__ import annotations

import numpy as np
import pandas as pd


def fill_tvt_from_input(horizontal_df: pd.DataFrame) -> pd.Series:
    """Predict TVT from known TVT_input with smooth interpolation and edge extrapolation."""
    if "TVT_input" not in horizontal_df.columns:
        return _fallback_from_z(horizontal_df)

    tvt = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce")
    if tvt.notna().sum() == 0:
        return _fallback_from_z(horizontal_df)

    x = np.arange(len(horizontal_df), dtype=float)
    known_mask = tvt.notna().to_numpy()
    known_x = x[known_mask]
    known_y = tvt.to_numpy(dtype=float)[known_mask]

    filled = np.interp(x, known_x, known_y)
    filled = _linear_edge_extrapolate(filled, known_x, known_y)
    return pd.Series(filled, index=horizontal_df.index, name="tvt")


def fill_tvt_flat_from_last_known(horizontal_df: pd.DataFrame) -> pd.Series:
    """Predict TVT by carrying the last known TVT_input through the evaluation zone."""
    if "TVT_input" not in horizontal_df.columns:
        return _fallback_from_z(horizontal_df)

    tvt = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce")
    if tvt.notna().sum() == 0:
        return _fallback_from_z(horizontal_df)

    filled = tvt.ffill().bfill()
    return filled.rename("tvt")


def fill_tvt(horizontal_df: pd.DataFrame, strategy: str = "flat") -> pd.Series:
    if strategy == "flat":
        return fill_tvt_flat_from_last_known(horizontal_df)
    if strategy == "interpolation":
        return fill_tvt_from_input(horizontal_df)
    raise ValueError(f"Unknown TVT fill strategy: {strategy}")


def predict_required_rows(horizontal_df: pd.DataFrame, row_indices: list[int]) -> pd.DataFrame:
    predictions = fill_tvt_flat_from_last_known(horizontal_df)
    valid_indices = [idx for idx in row_indices if 0 <= idx < len(horizontal_df)]
    return pd.DataFrame({"row_index": valid_indices, "tvt": predictions.iloc[valid_indices].to_numpy()})


def _linear_edge_extrapolate(filled: np.ndarray, known_x: np.ndarray, known_y: np.ndarray) -> np.ndarray:
    out = filled.copy()
    if len(known_x) < 2:
        return out

    left_slope = (known_y[1] - known_y[0]) / max(known_x[1] - known_x[0], 1.0)
    right_slope = (known_y[-1] - known_y[-2]) / max(known_x[-1] - known_x[-2], 1.0)

    left_mask = np.arange(len(out)) < known_x[0]
    right_mask = np.arange(len(out)) > known_x[-1]
    out[left_mask] = known_y[0] + left_slope * (np.arange(len(out))[left_mask] - known_x[0])
    out[right_mask] = known_y[-1] + right_slope * (np.arange(len(out))[right_mask] - known_x[-1])
    return out


def _fallback_from_z(horizontal_df: pd.DataFrame) -> pd.Series:
    if "Z" in horizontal_df.columns:
        z = pd.to_numeric(horizontal_df["Z"], errors="coerce")
        fallback = z.interpolate(limit_direction="both").bfill().ffill()
        if fallback.notna().any():
            return fallback.rename("tvt")

    return pd.Series(np.zeros(len(horizontal_df), dtype=float), index=horizontal_df.index, name="tvt")
