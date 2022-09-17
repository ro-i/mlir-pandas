from typing import Any, Dict
import unittest

import pandas as pd

from mlir_pandas import DataFrame, Series


class SetitemTest(unittest.TestCase):
    data: Dict[str, Any] = {"col1": [1, 2, 3], "col2": [4, 5, 6]}

    def test_setitem_supported(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)

        s: Series = Series([4, 5, 6], name="series")
        s_orig: pd.Series = pd.Series([4, 5, 6], name="series_orig")

        df["col3"] = s_orig
        df_orig["col3"] = s_orig
        self.assertTrue(df_orig.equals(df._mock))

        df["col4"] = s
        df_orig["col4"] = s
        self.assertTrue(df_orig.equals(df._mock))
