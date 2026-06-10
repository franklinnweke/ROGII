from __future__ import annotations

import numpy as np
import pandas as pd

from rogii_geology.baseline import fill_tvt_from_input


def mask_contiguous_zone(
    df: pd.DataFrame,
    start_fraction: float = 0.65,
    zone_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series]:
    """Mask a contiguous TVT_input interval to simulate Kaggle's hidden evaluation zone."""
    if "TVT" not in df.columns:
        raise ValueError("Validation requires a TVT column.")

    n_rows = len(df)
    start = int(np.clip(start_fraction, 0.0, 0.95) * n_rows)
    width = max(1, int(np.clip(zone_fraction, 0.01, 0.8) * n_rows))
    end = min(n_rows, start + width)

    masked = df.copy()
    if "TVT_input" not in masked.columns:
        masked["TVT_input"] = masked["TVT"]
    masked.loc[masked.index[start:end], "TVT_input"] = np.nan
    target = df.loc[df.index[start:end], "TVT"].astype(float)
    return masked, target


def score_interpolation_baseline(df: pd.DataFrame) -> float:
    masked, target = mask_contiguous_zone(df)
    pred = fill_tvt_from_input(masked).loc[target.index]
    residual = target.to_numpy(dtype=float) - pred.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(residual**2)))
