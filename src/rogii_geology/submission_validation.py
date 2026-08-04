from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii_geology.io import discover_wells, read_sample_submission

REQUIRED_COLUMNS = ["id", "tvt"]


def validate_submission(data_dir: Path, submission_path: Path) -> dict[str, object]:
    """Validate a submission against the sample and test-well row bounds."""
    data_dir = Path(data_dir)
    submission_path = Path(submission_path)
    sample = read_sample_submission(data_dir)
    submission = pd.read_csv(submission_path)

    if submission.columns.tolist() != REQUIRED_COLUMNS:
        raise ValueError(
            f"Submission columns must be exactly {REQUIRED_COLUMNS}; "
            f"found {submission.columns.tolist()}"
        )
    if len(submission) != len(sample):
        raise ValueError(
            f"Submission row count must be {len(sample)}; found {len(submission)}"
        )
    if submission["id"].duplicated().any():
        duplicates = submission.loc[submission["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"Submission contains duplicate ids: {duplicates}")

    actual_ids = submission["id"].astype(str)
    expected_ids = sample["id"].astype(str)
    if not actual_ids.equals(expected_ids):
        mismatch = int(np.flatnonzero(actual_ids.to_numpy() != expected_ids.to_numpy())[0])
        raise ValueError(
            "Submission ids must match sample_submission.csv in exact order; "
            f"first mismatch at row {mismatch}"
        )

    if not pd.api.types.is_numeric_dtype(submission["tvt"]):
        raise ValueError("Submission tvt values must be numeric")
    tvt = submission["tvt"].to_numpy(dtype=float)
    if not np.isfinite(tvt).all():
        bad_rows = np.flatnonzero(~np.isfinite(tvt))[:5].tolist()
        raise ValueError(f"Submission tvt values contain NaN or infinity at rows {bad_rows}")

    wells = {well.well_id: well for well in discover_wells(data_dir / "test")}
    rows_by_well: dict[str, int] = {}
    for value in actual_ids:
        well_id, separator, row_text = value.rpartition("_")
        if not separator or not row_text.isdigit():
            raise ValueError(f"Invalid submission id format: {value}")
        if well_id not in wells:
            raise ValueError(f"Submission id references missing test well: {well_id}")

        row_index = int(row_text)
        if well_id not in rows_by_well:
            rows_by_well[well_id] = _csv_data_row_count(wells[well_id].horizontal)
        if not 0 <= row_index < rows_by_well[well_id]:
            raise ValueError(
                f"Submission id {value} is outside test-well row bounds "
                f"0..{rows_by_well[well_id] - 1}"
            )

    return {
        "submission_path": str(submission_path),
        "columns": REQUIRED_COLUMNS,
        "rows": len(submission),
        "wells": len(rows_by_well),
        "id_order_matches_sample": True,
        "duplicate_ids": 0,
        "finite_numeric_tvt": True,
        "test_row_bounds_valid": True,
    }


def _csv_data_row_count(path: Path) -> int:
    return len(pd.read_csv(path, usecols=["TVT_input"]))
