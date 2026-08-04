from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from rogii_geology.baseline import fill_tvt, fill_tvt_saturated_ramp
from rogii_geology.io import discover_wells, read_horizontal, read_sample_submission


def parse_submission_ids(sample_submission: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for value in sample_submission["id"].astype(str):
        well_id, row_index = value.rsplit("_", 1)
        groups.setdefault(well_id, []).append(int(row_index))
    return groups


def make_baseline_submission(
    data_dir: Path,
    strategy: str = "saturated_ramp",
    *,
    collect_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    data_dir = Path(data_dir)
    sample = read_sample_submission(data_dir)
    requested = parse_submission_ids(sample)
    wells = {well.well_id: well for well in discover_wells(data_dir / "test")}

    predictions: dict[str, float] = {}
    ramp_applied = 0
    fallback_reasons: Counter[str] = Counter()

    for well_id, row_indices in requested.items():
        if well_id not in wells:
            raise FileNotFoundError(f"Missing horizontal well file for {well_id}")
        horizontal = read_horizontal(wells[well_id].horizontal)
        n_rows = len(horizontal)

        meta: dict[str, object] = {}
        if strategy == "saturated_ramp":
            tvt = fill_tvt_saturated_ramp(horizontal, meta=meta)
            if meta.get("used_ramp"):
                ramp_applied += 1
            else:
                reason = str(meta.get("fallback_reason") or "unknown")
                fallback_reasons[reason] += 1
        else:
            tvt = fill_tvt(horizontal, strategy=strategy)

        for row_index in row_indices:
            if not 0 <= row_index < n_rows:
                raise ValueError(
                    f"Submission id {well_id}_{row_index} is outside test-well row bounds "
                    f"0..{n_rows - 1}"
                )
            predictions[f"{well_id}_{row_index}"] = float(tvt.iloc[row_index])

    out = sample[["id"]].copy()
    out["tvt"] = out["id"].map(predictions)
    if out["tvt"].isna().any():
        missing = out.loc[out["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Missing predictions for sample ids: {missing}")
    if not pd.api.types.is_numeric_dtype(out["tvt"]):
        raise ValueError("Submission tvt values must be numeric")
    tvt_values = out["tvt"].to_numpy(dtype=float)
    if not np.isfinite(tvt_values).all():
        bad_rows = np.flatnonzero(~np.isfinite(tvt_values))[:5].tolist()
        raise ValueError(f"Submission tvt values contain NaN or infinity at rows {bad_rows}")

    run_meta: dict[str, object] = {
        "strategy": strategy,
        "wells": len(requested),
        "ramp_applied": ramp_applied if strategy == "saturated_ramp" else None,
        "fallback_reasons": dict(fallback_reasons) if strategy == "saturated_ramp" else {},
    }
    if collect_meta:
        return out, run_meta
    return out


def write_submission(
    data_dir: Path,
    output_path: Path,
    strategy: str = "saturated_ramp",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission, run_meta = make_baseline_submission(
        data_dir, strategy=strategy, collect_meta=True
    )
    submission.to_csv(output_path, index=False)
    print(
        f"strategy={run_meta['strategy']} wells={run_meta['wells']} "
        f"ramp_applied={run_meta['ramp_applied']} "
        f"fallback_reasons={run_meta['fallback_reasons']}"
    )
    return output_path
