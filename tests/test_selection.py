from typing import Any, Dict, cast
import unittest

import pandas as pd

from mlir_pandas import DataFrame, Series


class SelectionTest(unittest.TestCase):
    # Discard (reset) index. (TODO?)
    data: Dict[str, Any] = {"col1": [1, 2, 3], "col2": [1, 5, 2]}

    def test_selection_simple(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)
        s: Series = df[df["col1"] > 2]
        s_orig: pd.Series = df_orig[df_orig["col1"] > 2]
        s = cast(Series, s.reset_index(drop=True))
        s_orig = cast(pd.Series, s_orig.reset_index(drop=True))
        self.assertTrue(s_orig.equals(s._mock))

    def test_selection_multiple_columns(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)
        df = df[df["col1"] >= df["col2"]].reset_index(drop=True)
        df_orig = df_orig[df_orig["col1"] >= df_orig["col2"]].reset_index(drop=True)
        self.assertTrue(df_orig.equals(df._mock))

    def test_selection_complex(self) -> None:
        data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        df: DataFrame = DataFrame(data)
        df_orig: pd.DataFrame = pd.DataFrame(data)
        df[(df["col1"] <= 2) & (df["col2"] >= 5)]
        df_orig[(df_orig["col1"] <= 2) & (df_orig["col2"] >= 5)]
        self.assertTrue(df_orig.equals(df._mock))
