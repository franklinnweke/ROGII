import unittest

import numpy as np
import pandas as pd

from rogii_geology.baseline import fill_tvt_from_input
from rogii_geology.submission import parse_submission_ids
from rogii_geology.validation import score_interpolation_baseline


class BaselineTests(unittest.TestCase):
    def test_fill_tvt_interpolates_missing_zone(self) -> None:
        df = pd.DataFrame({"TVT_input": [100.0, 101.0, np.nan, np.nan, 104.0]})

        pred = fill_tvt_from_input(df)

        self.assertEqual(pred.tolist(), [100.0, 101.0, 102.0, 103.0, 104.0])

    def test_parse_submission_ids_groups_by_well(self) -> None:
        sample = pd.DataFrame({"id": ["000d7d20_1442", "000d7d20_1443", "abc12345_10"]})

        parsed = parse_submission_ids(sample)

        self.assertEqual(parsed, {"000d7d20": [1442, 1443], "abc12345": [10]})

    def test_validation_scores_synthetic_linear_tvt(self) -> None:
        df = pd.DataFrame({"TVT": np.arange(100.0, 200.0), "TVT_input": np.arange(100.0, 200.0)})

        rmse = score_interpolation_baseline(df)

        self.assertEqual(rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
