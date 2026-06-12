import unittest

import numpy as np
import pandas as pd

from rogii_geology.correlation import (
    blend_interpolation_and_correlation,
    estimate_tvt_from_typewell_gr,
)


class CorrelationTests(unittest.TestCase):
    def test_blend_keeps_interpolation_when_correlation_missing(self) -> None:
        interpolation = pd.Series([10.0, 20.0], index=[1, 2])
        correlation = pd.Series([np.nan, 30.0], index=[1, 2])

        blended = blend_interpolation_and_correlation(interpolation, correlation, 0.5)

        self.assertEqual(blended.loc[1], 10.0)
        self.assertEqual(blended.loc[2], 25.0)

    def test_estimate_tvt_from_typewell_gr_returns_requested_rows(self) -> None:
        tvt = np.arange(100.0, 200.0)
        gr = np.sin(np.linspace(0.0, 8.0, len(tvt))) * 20.0 + 80.0
        typewell = pd.DataFrame({"TVT": tvt, "GR": gr})
        horizontal = pd.DataFrame(
            {
                "GR": gr.copy(),
                "TVT_input": tvt.copy(),
            }
        )
        horizontal.loc[40:50, "TVT_input"] = np.nan

        pred = estimate_tvt_from_typewell_gr(horizontal, typewell, [45], window=21, search_radius=30)

        self.assertEqual(list(pred.index), [45])
        self.assertTrue(abs(float(pred.iloc[0]) - 145.0) <= 2.0)


if __name__ == "__main__":
    unittest.main()
