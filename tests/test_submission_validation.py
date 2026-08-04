import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rogii_geology.submission_validation import validate_submission


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        test_dir = self.data_dir / "test"
        test_dir.mkdir()
        pd.DataFrame({"id": ["well0001_1", "well0001_2"], "tvt": [0.0, 0.0]}).to_csv(
            self.data_dir / "sample_submission.csv", index=False
        )
        pd.DataFrame({"TVT_input": [1.0, np.nan, np.nan]}).to_csv(
            test_dir / "well0001__horizontal_well.csv", index=False
        )
        self.submission_path = self.data_dir / "submission.csv"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_submission(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.submission_path, index=False)

    def test_accepts_compliant_submission(self) -> None:
        self.write_submission(
            pd.DataFrame({"id": ["well0001_1", "well0001_2"], "tvt": [10.0, 11.0]})
        )

        report = validate_submission(self.data_dir, self.submission_path)

        self.assertEqual(report["rows"], 2)
        self.assertTrue(report["test_row_bounds_valid"])

    def test_rejects_wrong_column_order(self) -> None:
        self.write_submission(
            pd.DataFrame({"tvt": [10.0, 11.0], "id": ["well0001_1", "well0001_2"]})
        )

        with self.assertRaisesRegex(ValueError, "columns must be exactly"):
            validate_submission(self.data_dir, self.submission_path)

    def test_rejects_duplicate_or_reordered_ids(self) -> None:
        self.write_submission(
            pd.DataFrame({"id": ["well0001_1", "well0001_1"], "tvt": [10.0, 11.0]})
        )

        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            validate_submission(self.data_dir, self.submission_path)

    def test_rejects_non_finite_tvt(self) -> None:
        self.write_submission(
            pd.DataFrame({"id": ["well0001_1", "well0001_2"], "tvt": [10.0, np.inf]})
        )

        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            validate_submission(self.data_dir, self.submission_path)

    def test_rejects_out_of_bounds_row_suffix(self) -> None:
        sample = pd.DataFrame({"id": ["well0001_1", "well0001_3"], "tvt": [0.0, 0.0]})
        sample.to_csv(self.data_dir / "sample_submission.csv", index=False)
        self.write_submission(sample.assign(tvt=[10.0, 11.0]))

        with self.assertRaisesRegex(ValueError, "outside test-well row bounds"):
            validate_submission(self.data_dir, self.submission_path)


if __name__ == "__main__":
    unittest.main()
