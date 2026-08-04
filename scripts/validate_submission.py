from __future__ import annotations

import argparse
import json
from pathlib import Path

from rogii_geology.submission_validation import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a ROGII Kaggle submission.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    report = validate_submission(args.data_dir, args.submission)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
