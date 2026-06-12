from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_tvt_from_typewell_gr(
    horizontal_df: pd.DataFrame,
    typewell_df: pd.DataFrame,
    row_indices: list[int] | np.ndarray,
    window: int = 41,
    search_radius: float = 300.0,
    candidate_step: int = 5,
) -> pd.Series:
    """Estimate TVT by matching local horizontal GR windows to the typewell GR curve.

    This is intentionally simple and deterministic. For each row, it uses the current
    baseline TVT estimate as a center point, then searches nearby typewell TVT values
    for the local gamma-ray pattern with the lowest normalized squared error.
    """
    if "GR" not in horizontal_df.columns or "GR" not in typewell_df.columns or "TVT" not in typewell_df.columns:
        raise ValueError("Horizontal GR, typewell GR, and typewell TVT are required.")

    row_indices = np.asarray(row_indices, dtype=int)
    if len(row_indices) == 0:
        return pd.Series(dtype=float, name="tvt")

    horizontal_gr = pd.to_numeric(horizontal_df["GR"], errors="coerce").interpolate(
        limit_direction="both"
    )
    typewell = (
        typewell_df[["TVT", "GR"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .sort_values("TVT")
        .drop_duplicates("TVT")
    )
    if typewell.empty or horizontal_gr.isna().all():
        return pd.Series(np.nan, index=row_indices, name="tvt")

    type_tvt = typewell["TVT"].to_numpy(dtype=float)
    type_gr = typewell["GR"].to_numpy(dtype=float)

    center_estimate = _center_estimate(horizontal_df)
    predictions = []
    half_window = max(1, window // 2)

    for row_idx in row_indices:
        left = max(0, row_idx - half_window)
        right = min(len(horizontal_df), row_idx + half_window + 1)
        local_gr = horizontal_gr.iloc[left:right].to_numpy(dtype=float)
        local_offsets = np.arange(left, right, dtype=float) - float(row_idx)

        if not np.isfinite(local_gr).any():
            predictions.append(np.nan)
            continue

        center = float(center_estimate.iloc[row_idx])
        candidate_mask = (type_tvt >= center - search_radius) & (type_tvt <= center + search_radius)
        if candidate_mask.sum() < 5:
            candidate_mask = np.ones_like(type_tvt, dtype=bool)

        candidate_tvt = type_tvt[candidate_mask]
        step = max(1, int(candidate_step))
        candidate_tvt = candidate_tvt[::step]
        if len(candidate_tvt) == 0:
            candidate_tvt = type_tvt[::step]
        best_tvt = _best_typewell_match(candidate_tvt, type_tvt, type_gr, local_offsets, local_gr)
        predictions.append(best_tvt)

    return pd.Series(predictions, index=row_indices, name="tvt")


def blend_interpolation_and_correlation(
    interpolation: pd.Series,
    correlation: pd.Series,
    correlation_weight: float = 0.35,
) -> pd.Series:
    aligned_corr = correlation.reindex(interpolation.index)
    weight = float(np.clip(correlation_weight, 0.0, 1.0))
    blended = interpolation.copy().astype(float)
    valid = aligned_corr.notna()
    blended.loc[valid] = (1.0 - weight) * blended.loc[valid] + weight * aligned_corr.loc[valid]
    return blended.rename("tvt")


def _center_estimate(horizontal_df: pd.DataFrame) -> pd.Series:
    if "TVT_input" in horizontal_df.columns:
        tvt = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce")
        if tvt.notna().any():
            return tvt.interpolate(limit_direction="both").bfill().ffill()
    if "Z" in horizontal_df.columns:
        return pd.to_numeric(horizontal_df["Z"], errors="coerce").interpolate(
            limit_direction="both"
        ).bfill().ffill()
    return pd.Series(np.zeros(len(horizontal_df), dtype=float), index=horizontal_df.index)


def _best_typewell_match(
    candidate_tvt: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    local_offsets: np.ndarray,
    local_gr: np.ndarray,
) -> float:
    local = _standardize(local_gr)
    best_score = np.inf
    best_tvt = float(candidate_tvt[0])

    for tvt in candidate_tvt:
        sample_tvt = tvt + local_offsets
        type_sample = np.interp(sample_tvt, type_tvt, type_gr)
        score = float(np.nanmean((_standardize(type_sample) - local) ** 2))
        if score < best_score:
            best_score = score
            best_tvt = float(tvt)

    return best_tvt


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-6:
        return values - mean
    return (values - mean) / std
