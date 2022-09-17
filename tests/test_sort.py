from typing import Any, Dict, Tuple, cast
import unittest

import pandas as pd

from mlir_pandas import DataFrame, Series


class SortTest(unittest.TestCase):
    data: Dict[str, Any] = {"col1": [1, 3, 2, 2], "col2": [6, 5, 3, 4]}
    data_s: Tuple[str, Any] = ("series", [1, 3, 2, 2])

    def test_sort_df(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)
        dfs: DataFrame = cast(DataFrame, df.sort_values("col1", ignore_index=True))
        dfs_orig: pd.DataFrame = cast(pd.DataFrame, df_orig.sort_values("col1", ignore_index=True))
        self.assertTrue(dfs_orig.equals(dfs._mock))
        dfs = cast(DataFrame, df.sort_values(
            ["col1", "col2"], ascending=[True, False], ignore_index=True
        ))
        dfs_orig = cast(pd.DataFrame, df_orig.sort_values(
            ["col1", "col2"], ascending=[True, False], ignore_index=True
        ))
        self.assertTrue(dfs_orig.equals(dfs._mock))

    def test_sort_series(self) -> None:
        s: Series = Series(self.data_s[1], name=self.data_s[0])
        s_orig: pd.Series = pd.Series(self.data_s[1], name=self.data_s[0])
        self.assertTrue(
            cast(pd.Series, s_orig.sort_values(ignore_index=True)).equals(
                cast(Series, s.sort_values(ignore_index=True))._mock
            )
        )
        self.assertTrue(
            cast(pd.Series, s_orig.sort_values(ascending=False, ignore_index=True)).equals(
                cast(Series, s.sort_values(ascending=False, ignore_index=True))._mock
            )
        )
