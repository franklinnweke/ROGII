import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rogii_geology.eda import build_target_gap_summary, write_eda_markdown


class EdaTests(unittest.TestCase):
    def test_build_target_gap_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            pd.DataFrame(
                {
                    "id": ["well0001_3", "well0001_4", "well0002_10"],
                    "tvt": [0.0, 0.0, 0.0],
                }
            ).to_csv(data_dir / "sample_submission.csv", index=False)

            summary = build_target_gap_summary(data_dir)

        self.assertEqual(summary.to_dict("records")[0]["well_id"], "well0001")
        self.assertEqual(summary.to_dict("records")[0]["prediction_rows"], 2)
        self.assertEqual(summary.to_dict("records")[0]["first_row"], 3)
        self.assertEqual(summary.to_dict("records")[0]["last_row"], 4)

    def test_write_eda_markdown_includes_modeling_implications(self) -> None:
        train = pd.DataFrame(
            {
                "rows": [10],
                "typewell_rows": [5],
                "tvt_input_missing": [2],
            }
        )
        test = pd.DataFrame(
            {
                "rows": [8],
                "typewell_rows": [4],
                "tvt_input_missing": [3],
            }
        )
        gaps = pd.DataFrame(
            {
                "well_id": ["well0001"],
                "prediction_rows": [3],
                "first_row": [5],
                "last_row": [7],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "eda.md"
            write_eda_markdown(train, test, gaps, output_path)
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("ROGII EDA Summary", text)
        self.assertIn("Practical Modeling Implications", text)
        self.assertIn("well0001", text)


if __name__ == "__main__":
    unittest.main()
