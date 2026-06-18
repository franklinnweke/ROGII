from __future__ import annotations

import argparse
from pathlib import Path

from rogii_geology.submission import write_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple ROGII baseline submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument("--strategy", choices=["flat", "interpolation"], default="flat")
    args = parser.parse_args()

    output_path = write_submission(args.data_dir, args.output, strategy=args.strategy)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
