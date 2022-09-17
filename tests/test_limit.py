from typing import Any, Dict, Tuple
import unittest

import pandas as pd

from mlir_pandas import DataFrame, Series


class LimitTest(unittest.TestCase):
    data: Dict[str, Any] = {"col1": [1, 3, 2, 2], "col2": [6, 5, 3, 4]}
    data_s: Tuple[str, Any] = ("series", [1, 3, 2, 2])

    def test_limit_df(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)
        self.assertTrue(df_orig.head(3).equals(df.head(3)._mock))

    def test_limit_series(self) -> None:
        s: Series = Series(self.data_s[1], name=self.data_s[0])
        s_orig: pd.Series = pd.Series(self.data_s[1], name=self.data_s[0])
        self.assertTrue(s_orig.head(3).equals(s.head(3)._mock))
