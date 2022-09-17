from typing import Any, Dict, List, Tuple
import unittest

import pandas as pd

from mlir_pandas import DataFrame, Series


class ArithColsTest(unittest.TestCase):
    data: Dict[str, Any] = {"col1": [1, 2, 3], "col2": [1, 5, 2]}
    data_s: List[Tuple[Tuple[str, Any], Tuple[str, Any]]] = [
        (("series", [1, 2, 3]), ("series", [4, 5, 6]))
    ]
    data_s_bool: List[Tuple[Tuple[str, Any], Tuple[str, Any]]] = [
        (("series", [True, False, True]), ("series", [True, False, False]))
    ]

    def test_arith_cols_series(self) -> None:
        for (name1, data1), (name2, data2) in self.data_s:
            s1: Series = Series(data1, name=name1)
            s1_orig: pd.Series = pd.Series(data1, name=name1)
            s2: Series = Series(data2, name=name2)
            s2_orig: pd.Series = pd.Series(data2, name=name2)
            self.assertTrue((s1_orig + s2_orig).equals((s1 + s2)._mock))
            self.assertTrue((s1_orig - s2_orig).equals((s1 - s2)._mock))
            self.assertTrue((s1_orig * s2_orig).equals((s1 * s2)._mock))

    def test_arith_bool_cols_series(self) -> None:
        for (name1, data1), (name2, data2) in self.data_s_bool:
            s1: Series = Series(data1, name=name1)
            s1_orig: pd.Series = pd.Series(data1, name=name1)
            s2: Series = Series(data2, name=name2)
            s2_orig: pd.Series = pd.Series(data2, name=name2)
            self.assertTrue((s1_orig & s2_orig).equals((s1 & s2)._mock))
            self.assertTrue((s1_orig | s2_orig).equals((s1 | s2)._mock))
            self.assertTrue((~s1_orig).equals((~s1)._mock))

    def test_arith_cols_simple(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)
        self.assertTrue(
            (df_orig["col1"] + df_orig["col2"]).equals((df["col1"] + df["col2"])._mock)
        )
