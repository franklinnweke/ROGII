from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rogii_geology.baseline import fill_tvt_from_input
from rogii_geology.io import discover_wells, read_horizontal
from rogii_geology.validation import mask_contiguous_zone


def evaluate_well(path: Path, start_fraction: float, zone_fraction: float) -> dict[str, float | int | str]:
    df = read_horizontal(path)
    masked, target = mask_contiguous_zone(df, start_fraction=start_fraction, zone_fraction=zone_fraction)
    prediction = fill_tvt_from_input(masked).loc[target.index]
    residual = target.to_numpy(dtype=float) - prediction.to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    return {
        "well_id": path.name.replace("__horizontal_well.csv", ""),
        "rows": len(df),
        "masked_rows": len(target),
        "rmse": rmse,
        "mae": mae,
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
    }


def evaluate_train(
    data_dir: Path,
    start_fraction: float,
    zone_fraction: float,
) -> pd.DataFrame:
    wells = discover_wells(Path(data_dir) / "train")
    rows = [evaluate_well(well.horizontal, start_fraction, zone_fraction) for well in wells]
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def write_markdown(results: pd.DataFrame, output_path: Path) -> None:
    total_squared_error = (results["rmse"] ** 2) * results["masked_rows"]
    weighted_rmse = float(np.sqrt(total_squared_error.sum() / results["masked_rows"].sum()))

    lines = [
        "# Baseline Validation",
        "",
        "Validation masks a contiguous interval inside each training well, then predicts that interval from `TVT_input` using the current interpolation/extrapolation baseline.",
        "",
        f"- Wells evaluated: {len(results)}",
        f"- Masked rows: {int(results['masked_rows'].sum())}",
        f"- Weighted RMSE: {weighted_rmse:.4f}",
        f"- Median per-well RMSE: {results['rmse'].median():.4f}",
        f"- Mean per-well RMSE: {results['rmse'].mean():.4f}",
        f"- Worst per-well RMSE: {results['rmse'].max():.4f}",
        "",
        "## Worst Wells",
        "",
        "| Well | Rows | Masked Rows | RMSE | MAE | Target Range | Prediction Range |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for _, row in results.head(10).iterrows():
        lines.append(
            "| {well} | {rows} | {masked} | {rmse:.4f} | {mae:.4f} | "
            "{target_min:.2f}-{target_max:.2f} | {prediction_min:.2f}-{prediction_max:.2f} |".format(
                well=row["well_id"],
                rows=int(row["rows"]),
                masked=int(row["masked_rows"]),
                rmse=float(row["rmse"]),
                mae=float(row["mae"]),
                target_min=float(row["target_min"]),
                target_max=float(row["target_max"]),
                prediction_min=float(row["prediction_min"]),
                prediction_max=float(row["prediction_max"]),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the interpolation baseline on train wells.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--start-fraction", type=float, default=0.65)
    parser.add_argument("--zone-fraction", type=float, default=0.20)
    parser.add_argument("--csv-output", type=Path, default=Path("work/baseline_validation.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("outputs/baseline_validation.md"))
    args = parser.parse_args()

    results = evaluate_train(args.data_dir, args.start_fraction, args.zone_fraction)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.csv_output, index=False)
    write_markdown(results, args.markdown_output)

    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
