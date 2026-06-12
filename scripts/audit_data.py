from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from rogii_geology.io import discover_wells
from rogii_geology.submission import parse_submission_ids


def csv_columns(path: Path) -> tuple[str, ...]:
    return tuple(pd.read_csv(path, nrows=0).columns)


def csv_row_count(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8")) - 1


def describe_counts(values: list[int]) -> dict[str, float | int]:
    series = pd.Series(values, dtype="float64")
    return {
        "min": int(series.min()),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "max": int(series.max()),
    }


def audit_split(split_dir: Path) -> dict[str, object]:
    wells = discover_wells(split_dir)
    horizontal_cols = Counter(csv_columns(well.horizontal) for well in wells)
    typewell_cols = Counter(csv_columns(well.typewell) for well in wells if well.typewell)
    horizontal_rows = [csv_row_count(well.horizontal) for well in wells]
    typewell_rows = [csv_row_count(well.typewell) for well in wells if well.typewell]

    return {
        "well_count": len(wells),
        "horizontal_file_count": len([well for well in wells if well.horizontal.exists()]),
        "typewell_file_count": len([well for well in wells if well.typewell]),
        "image_file_count": len([well for well in wells if well.image]),
        "horizontal_schema_variants": [
            {"count": count, "columns": list(columns)}
            for columns, count in horizontal_cols.most_common()
        ],
        "typewell_schema_variants": [
            {"count": count, "columns": list(columns)}
            for columns, count in typewell_cols.most_common()
        ],
        "horizontal_row_counts": describe_counts(horizontal_rows) if horizontal_rows else {},
        "typewell_row_counts": describe_counts(typewell_rows) if typewell_rows else {},
    }


def audit_data(data_dir: Path) -> dict[str, object]:
    data_dir = Path(data_dir)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    sample_groups = parse_submission_ids(sample)

    return {
        "data_dir": str(data_dir),
        "file_count": len([path for path in data_dir.rglob("*") if path.is_file()]),
        "sample_submission_rows": len(sample),
        "sample_submission_well_count": len(sample_groups),
        "sample_submission_rows_by_well": {
            well_id: len(row_indices) for well_id, row_indices in sample_groups.items()
        },
        "train": audit_split(data_dir / "train"),
        "test": audit_split(data_dir / "test"),
    }


def write_markdown(report: dict[str, object], output_path: Path) -> None:
    train = report["train"]
    test = report["test"]
    assert isinstance(train, dict)
    assert isinstance(test, dict)

    lines = [
        "# ROGII Data Audit",
        "",
        f"- Data directory: `{report['data_dir']}`",
        f"- Total files: {report['file_count']}",
        f"- Sample submission rows: {report['sample_submission_rows']}",
        f"- Sample submission wells: {report['sample_submission_well_count']}",
        "",
        "## Split Summary",
        "",
        "| Split | Wells | Horizontal CSVs | Typewell CSVs | PNGs | Horizontal Rows | Typewell Rows |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for name, split in (("train", train), ("test", test)):
        horizontal_counts = split["horizontal_row_counts"]
        typewell_counts = split["typewell_row_counts"]
        assert isinstance(horizontal_counts, dict)
        assert isinstance(typewell_counts, dict)
        lines.append(
            "| {name} | {wells} | {horizontal} | {typewell} | {images} | "
            "{h_min}-{h_max}, median {h_median:.1f} | {t_min}-{t_max}, median {t_median:.1f} |".format(
                name=name,
                wells=split["well_count"],
                horizontal=split["horizontal_file_count"],
                typewell=split["typewell_file_count"],
                images=split["image_file_count"],
                h_min=horizontal_counts.get("min", 0),
                h_max=horizontal_counts.get("max", 0),
                h_median=horizontal_counts.get("median", 0.0),
                t_min=typewell_counts.get("min", 0),
                t_max=typewell_counts.get("max", 0),
                t_median=typewell_counts.get("median", 0.0),
            )
        )

    lines.extend(["", "## Schema Variants", ""])
    for name, split in (("train", train), ("test", test)):
        lines.append(f"### {name}")
        for key in ("horizontal_schema_variants", "typewell_schema_variants"):
            variants = split[key]
            assert isinstance(variants, list)
            lines.append(f"- {key}: {len(variants)} variant(s)")
            for variant in variants[:3]:
                assert isinstance(variant, dict)
                lines.append(
                    f"  - {variant['count']} file(s): `{', '.join(variant['columns'])}`"
                )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local ROGII competition data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--json-output", type=Path, default=Path("work/data_audit.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("outputs/data_audit.md"))
    args = parser.parse_args()

    report = audit_data(args.data_dir)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.markdown_output)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
