from __future__ import annotations

import numpy as np
import pandas as pd


FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create trajectory and log features without using hidden target values."""
    out = df.copy()

    if "MD" in out.columns:
        md = out["MD"].astype(float)
        out["md_from_start"] = md - md.min()
        span = md.max() - md.min()
        out["md_fraction"] = 0.0 if span == 0 else (md - md.min()) / span

    for col in ("X", "Y", "Z", "GR"):
        if col in out.columns:
            series = out[col].astype(float)
            out[f"{col.lower()}_diff"] = series.diff().fillna(0.0)
            out[f"{col.lower()}_roll_mean_7"] = (
                series.rolling(window=7, center=True, min_periods=1).mean()
            )
            out[f"{col.lower()}_roll_std_15"] = (
                series.rolling(window=15, center=True, min_periods=2).std().fillna(0.0)
            )

    if {"X", "Y"}.issubset(out.columns):
        dx = out["X"].astype(float).diff().fillna(0.0)
        dy = out["Y"].astype(float).diff().fillna(0.0)
        out["horizontal_step"] = np.sqrt(dx**2 + dy**2)
        out["horizontal_distance"] = out["horizontal_step"].cumsum()

    if "Z" in out.columns:
        for formation in FORMATION_COLUMNS:
            if formation in out.columns:
                out[f"z_minus_{formation.lower()}"] = out["Z"].astype(float) - out[
                    formation
                ].astype(float)

    return out
