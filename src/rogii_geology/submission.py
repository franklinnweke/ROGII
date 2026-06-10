from __future__ import annotations

from pathlib import Path

import pandas as pd

from rogii_geology.baseline import fill_tvt_from_input
from rogii_geology.io import discover_wells, read_horizontal, read_sample_submission


def parse_submission_ids(sample_submission: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for value in sample_submission["id"].astype(str):
        well_id, row_index = value.rsplit("_", 1)
        groups.setdefault(well_id, []).append(int(row_index))
    return groups


def make_baseline_submission(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    sample = read_sample_submission(data_dir)
    requested = parse_submission_ids(sample)
    wells = {well.well_id: well for well in discover_wells(data_dir / "test")}

    predictions: dict[str, float] = {}
    for well_id, row_indices in requested.items():
        if well_id not in wells:
            raise FileNotFoundError(f"Missing horizontal well file for {well_id}")
        horizontal = read_horizontal(wells[well_id].horizontal)
        tvt = fill_tvt_from_input(horizontal)
        for row_index in row_indices:
            predictions[f"{well_id}_{row_index}"] = float(tvt.iloc[row_index])

    out = sample[["id"]].copy()
    out["tvt"] = out["id"].map(predictions)
    if out["tvt"].isna().any():
        missing = out.loc[out["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Missing predictions for sample ids: {missing}")
    return out


def write_submission(data_dir: Path, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = make_baseline_submission(data_dir)
    submission.to_csv(output_path, index=False)
    return output_path
