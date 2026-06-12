from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rogii_geology.io import WellFiles, discover_wells, read_horizontal


FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


def summarize_horizontal_well(well: WellFiles) -> dict[str, float | int | str]:
    df = read_horizontal(well.horizontal)
    row: dict[str, float | int | str] = {
        "well_id": well.well_id,
        "rows": len(df),
        "has_typewell": int(well.typewell is not None),
        "has_image": int(well.image is not None),
    }

    for col in ("MD", "X", "Y", "Z", "GR", "TVT", "TVT_input"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            row[f"{col.lower()}_min"] = float(values.min())
            row[f"{col.lower()}_median"] = float(values.median())
            row[f"{col.lower()}_max"] = float(values.max())
            row[f"{col.lower()}_missing"] = int(values.isna().sum())

    if {"X", "Y"}.issubset(df.columns):
        dx = pd.to_numeric(df["X"], errors="coerce").diff().fillna(0.0)
        dy = pd.to_numeric(df["Y"], errors="coerce").diff().fillna(0.0)
        row["horizontal_distance"] = float(np.sqrt(dx**2 + dy**2).sum())

    if "MD" in df.columns:
        md = pd.to_numeric(df["MD"], errors="coerce")
        row["md_span"] = float(md.max() - md.min())

    if "Z" in df.columns:
        z = pd.to_numeric(df["Z"], errors="coerce")
        row["z_span"] = float(z.max() - z.min())

    for formation in FORMATION_COLUMNS:
        if formation in df.columns:
            row[f"{formation.lower()}_missing"] = int(df[formation].isna().sum())

    return row


def summarize_typewell(well: WellFiles) -> dict[str, float | int | str]:
    if well.typewell is None:
        return {"well_id": well.well_id, "typewell_rows": 0}

    df = pd.read_csv(well.typewell)
    row: dict[str, float | int | str] = {"well_id": well.well_id, "typewell_rows": len(df)}
    for col in ("TVT", "GR"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            row[f"typewell_{col.lower()}_min"] = float(values.min())
            row[f"typewell_{col.lower()}_median"] = float(values.median())
            row[f"typewell_{col.lower()}_max"] = float(values.max())
            row[f"typewell_{col.lower()}_missing"] = int(values.isna().sum())

    if "Geology" in df.columns:
        row["geology_units"] = int(df["Geology"].nunique(dropna=True))
        top_units = df["Geology"].value_counts(dropna=True).head(5)
        row["top_geology_units"] = "; ".join(f"{idx}:{count}" for idx, count in top_units.items())

    return row


def build_well_summary(data_dir: Path, split: str) -> pd.DataFrame:
    wells = discover_wells(Path(data_dir) / split)
    rows = []
    for well in wells:
        rows.append({**summarize_horizontal_well(well), **summarize_typewell(well)})
    return pd.DataFrame(rows)


def build_target_gap_summary(data_dir: Path) -> pd.DataFrame:
    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv")
    ids = sample["id"].astype(str).str.rsplit("_", n=1, expand=True)
    ids.columns = ["well_id", "row_index"]
    ids["row_index"] = ids["row_index"].astype(int)
    return (
        ids.groupby("well_id")
        .agg(
            prediction_rows=("row_index", "size"),
            first_row=("row_index", "min"),
            last_row=("row_index", "max"),
        )
        .reset_index()
    )


def write_eda_markdown(
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    target_gap_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# ROGII EDA Summary",
        "",
        "This report summarizes the local Kaggle data without redistributing raw competition files.",
        "",
        "## Dataset Shape",
        "",
        f"- Train wells: {len(train_summary)}",
        f"- Public example test wells: {len(test_summary)}",
        f"- Public sample-submission wells: {len(target_gap_summary)}",
        f"- Public sample-submission rows: {int(target_gap_summary['prediction_rows'].sum())}",
        "",
        "## Row Count Ranges",
        "",
        "| Split | Horizontal rows min | Median | Max | Typewell rows min | Median | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split, summary in (("train", train_summary), ("test", test_summary)):
        lines.append(
            "| {split} | {hmin:.0f} | {hmed:.0f} | {hmax:.0f} | {tmin:.0f} | {tmed:.0f} | {tmax:.0f} |".format(
                split=split,
                hmin=summary["rows"].min(),
                hmed=summary["rows"].median(),
                hmax=summary["rows"].max(),
                tmin=summary["typewell_rows"].min(),
                tmed=summary["typewell_rows"].median(),
                tmax=summary["typewell_rows"].max(),
            )
        )

    train_tvt_missing = int(train_summary.get("tvt_input_missing", pd.Series(dtype=int)).sum())
    train_rows = int(train_summary["rows"].sum())
    test_tvt_missing = int(test_summary.get("tvt_input_missing", pd.Series(dtype=int)).sum())
    test_rows = int(test_summary["rows"].sum())

    lines.extend(
        [
            "",
            "## Target Availability",
            "",
            f"- Train `TVT_input` missing rows: {train_tvt_missing:,} of {train_rows:,}",
            f"- Public test `TVT_input` missing rows: {test_tvt_missing:,} of {test_rows:,}",
            "- The hidden Kaggle test set replaces the public example test folder during notebook reruns.",
            "",
            "## Public Prediction Zones",
            "",
            "| Well | Prediction rows | First row | Last row |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for _, row in target_gap_summary.iterrows():
        lines.append(
            f"| {row['well_id']} | {int(row['prediction_rows'])} | {int(row['first_row'])} | {int(row['last_row'])} |"
        )

    lines.extend(
        [
            "",
            "## Practical Modeling Implications",
            "",
            "- Train and public test schemas differ: formation-surface columns and `Geology` labels are present in train but absent from public test horizontal/typewell files.",
            "- A submission notebook must not rely on train-only columns for test inference unless those columns are engineered from files available at inference time.",
            "- Validation should mask contiguous intervals in train wells and should also hold out entire wells.",
            "- The current interpolation baseline is a valid submission path, but validation shows it fails when geological position shifts away from the local continuity assumption.",
            "- Next useful features should focus on gamma-ray pattern matching between horizontal wells and typewells, trajectory geometry, and smooth residual modeling.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
