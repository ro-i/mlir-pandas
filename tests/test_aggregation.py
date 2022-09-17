from typing import Any, Dict
import unittest

import pandas as pd

from mlir_pandas import DataFrame, Series


class AggregationTest(unittest.TestCase):
    data: Dict[str, Any] = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}

    def test_aggregation_df(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)

        self.assertTrue(
            df_orig.agg(s=("col1", "sum"), m=("col2", "mean")).equals(
                df.agg(s=("col1", "sum"), m=("col2", "mean"))._mock
            )
        )

        self.assertTrue(
            df_orig.agg(s=("col1", "sum"), c=("col1", "count"), m=("col2", "mean")).equals(
                df.agg(s=("col1", "sum"), c=("col1", "count"), m=("col2", "mean"))._mock
            )
        )

        self.assertTrue(
            df_orig.agg(s=("col1", "sum")).equals(
                df.agg(s=("col1", "sum"))._mock
            )
        )

    def test_aggregation_series(self) -> None:
        s: Series = Series(self.data["col1"], name="col1")
        s_orig: pd.Series = pd.Series(self.data["col1"], name="col1")

        self.assertTrue(s_orig.agg(s="sum").astype("Int64").equals(s.agg(s="sum")._mock))

        self.assertTrue(
            s_orig.agg(s="sum", m="mean").astype("Int64").equals(s.agg(s="sum", m="mean")._mock)
        )
