from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WellFiles:
    well_id: str
    horizontal: Path
    typewell: Path | None = None
    image: Path | None = None


def discover_wells(split_dir: Path) -> list[WellFiles]:
    """Discover well files in a Kaggle train or test directory."""
    split_dir = Path(split_dir)
    horizontal_files = sorted(split_dir.glob("*__horizontal_well.csv"))
    wells: list[WellFiles] = []

    for horizontal in horizontal_files:
        well_id = horizontal.name.replace("__horizontal_well.csv", "")
        typewell = split_dir / f"{well_id}__typewell.csv"
        image = split_dir / f"{well_id}.png"
        wells.append(
            WellFiles(
                well_id=well_id,
                horizontal=horizontal,
                typewell=typewell if typewell.exists() else None,
                image=image if image.exists() else None,
            )
        )

    return wells


def read_horizontal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "WELLNAME" not in df.columns:
        well_id = Path(path).name.replace("__horizontal_well.csv", "")
        df.insert(0, "WELLNAME", well_id)
    return df


def read_sample_submission(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "sample_submission.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing sample submission: {path}")
    return pd.read_csv(path)
