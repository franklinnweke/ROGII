from __future__ import annotations

import argparse
from pathlib import Path

from rogii_geology.eda import build_target_gap_summary, build_well_summary, write_eda_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ROGII EDA summaries from local data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/eda"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    train_summary = build_well_summary(args.data_dir, "train")
    test_summary = build_well_summary(args.data_dir, "test")
    target_gap_summary = build_target_gap_summary(args.data_dir)

    train_summary.to_csv(args.work_dir / "train_well_summary.csv", index=False)
    test_summary.to_csv(args.work_dir / "test_well_summary.csv", index=False)
    target_gap_summary.to_csv(args.work_dir / "target_gap_summary.csv", index=False)
    write_eda_markdown(
        train_summary,
        test_summary,
        target_gap_summary,
        args.output_dir / "eda_summary.md",
    )

    print(f"Wrote {args.output_dir / 'eda_summary.md'}")
    print(f"Wrote {args.work_dir / 'train_well_summary.csv'}")
    print(f"Wrote {args.work_dir / 'test_well_summary.csv'}")


if __name__ == "__main__":
    main()
