from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Guards for saturated-ramp polyfit stability on short or near-constant MD spans.
MIN_MD_SPAN_FT = 1.0
MAX_ABS_SLOPE = 5.0  # |dTVT/dMD|; pathological slopes fall back to flat


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


def fill_tvt_saturated_ramp(
    horizontal_df: pd.DataFrame,
    tau: float = 100.0,
    center: float = 100.0,
    scale: float = 100.0,
    gain: float = 1.0,
    tail_rows: int = 80,
    min_md_span: float = MIN_MD_SPAN_FT,
    max_abs_slope: float = MAX_ABS_SLOPE,
    meta: dict[str, Any] | None = None,
) -> pd.Series:
    """Blend flat continuity with a damped recent TVT trend in the hidden tail.

    When ``meta`` is provided, it is updated with ``used_ramp`` (bool) and
    ``fallback_reason`` (str | None) so callers can record silent degradations.
    """
    flat = fill_tvt_flat_from_last_known(horizontal_df).astype(float)

    def _fallback(reason: str) -> pd.Series:
        if meta is not None:
            meta["used_ramp"] = False
            meta["fallback_reason"] = reason
        return flat.rename("tvt")

    if "TVT_input" not in horizontal_df.columns or "MD" not in horizontal_df.columns:
        return _fallback("missing_tvt_input_or_md")

    tvt = pd.to_numeric(horizontal_df["TVT_input"], errors="coerce")
    md = pd.to_numeric(horizontal_df["MD"], errors="coerce")
    known_positions = np.flatnonzero(tvt.notna().to_numpy())
    if len(known_positions) < 2:
        return _fallback("insufficient_known_tvt")

    last_known_position = int(known_positions[-1])
    tail_positions = known_positions[-max(2, int(tail_rows)) :]
    valid_tail = md.iloc[tail_positions].notna().to_numpy()
    tail_positions = tail_positions[valid_tail]
    if len(tail_positions) < 2:
        return _fallback("insufficient_tail_with_md")

    tail_md = md.iloc[tail_positions].to_numpy(dtype=float)
    tail_tvt = tvt.iloc[tail_positions].to_numpy(dtype=float)
    md_span = float(np.ptp(tail_md))
    if not np.isfinite(md_span) or md_span < float(min_md_span):
        return _fallback("md_span_too_small")

    slope = float(np.polyfit(tail_md, tail_tvt, 1)[0])
    if not np.isfinite(slope) or abs(slope) > float(max_abs_slope):
        return _fallback("slope_unstable_or_extreme")

    hidden_positions = np.arange(last_known_position + 1, len(horizontal_df))
    if len(hidden_positions) == 0:
        return _fallback("no_hidden_tail")

    distance = md.iloc[hidden_positions].to_numpy(dtype=float) - float(md.iloc[last_known_position])
    if not np.isfinite(distance).all():
        return _fallback("non_finite_md_distance")

    safe_tau = max(float(tau), 1e-6)
    safe_scale = max(float(scale), 1e-6)
    distance = np.maximum(distance, 0.0)
    last_tvt = float(tvt.iloc[last_known_position])
    ramp = last_tvt + slope * safe_tau * (1.0 - np.exp(-distance / safe_tau))
    ramp_weight = 1.0 / (1.0 + np.exp(-np.clip((distance - center) / safe_scale, -60, 60)))
    flat.iloc[hidden_positions] = last_tvt + float(gain) * ramp_weight * (ramp - last_tvt)

    if meta is not None:
        meta["used_ramp"] = True
        meta["fallback_reason"] = None
        meta["slope"] = slope
        meta["md_span"] = md_span

    result = flat.rename("tvt")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        return _fallback("non_finite_ramp_output")
    return result


def fill_tvt(horizontal_df: pd.DataFrame, strategy: str = "saturated_ramp") -> pd.Series:
    if strategy == "flat":
        return fill_tvt_flat_from_last_known(horizontal_df)
    if strategy == "interpolation":
        return fill_tvt_from_input(horizontal_df)
    if strategy == "saturated_ramp":
        return fill_tvt_saturated_ramp(horizontal_df)
    raise ValueError(f"Unknown TVT fill strategy: {strategy}")


def predict_required_rows(
    horizontal_df: pd.DataFrame,
    row_indices: list[int],
    strategy: str = "saturated_ramp",
) -> pd.DataFrame:
    predictions = fill_tvt(horizontal_df, strategy=strategy)
    n_rows = len(horizontal_df)
    valid_indices: list[int] = []
    values: list[float] = []
    for idx in row_indices:
        if not 0 <= idx < n_rows:
            raise ValueError(
                f"Requested row_index {idx} is outside well bounds 0..{n_rows - 1}"
            )
        valid_indices.append(idx)
        values.append(float(predictions.iloc[idx]))
    return pd.DataFrame({"row_index": valid_indices, "tvt": values})


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
