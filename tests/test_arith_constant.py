from typing import Any, Dict, List, Tuple, Union
import unittest

import pandas as pd

from mlir_pandas import DataFrame, Series


class ArithConstantTest(unittest.TestCase):
    data: List[Tuple[Union[List[Any], Dict[str, Any]], Any]] = [
        ({"col1": [1, 2, 3]}, 1),
        ({"col1": [1, 2, 3], "col2": [4, 5, 6]}, 3),
        ({"col1": [1, 2, 3], "col2": [4, 5, 6]}, 1.5),
        ({"col1": [1, 2.5, 3]}, 1),
        ({"col1": [1, 2, 3], "col2": [4, 5.5, 6]}, 1.5)
    ]
    data_s: List[Tuple[str, Any, int]] = [
        ("series", [1, 2, 3, 5, 8, 13], 3)
    ]

    def test_arith_unsupported(self) -> None:
        for data, constant in self.data:
            df: DataFrame = DataFrame(data)
            df_orig: pd.DataFrame = pd.DataFrame(data)
            self.assertTrue((df_orig // constant).equals((df // constant)._mock))

    def test_constant_empty(self) -> None:
        self.assertTrue((pd.DataFrame() + 1).equals((DataFrame() + 1)._mock))

    def test_constant_supported(self) -> None:
        for data, constant in self.data:
            df: DataFrame = DataFrame(data)
            df_orig: pd.DataFrame = pd.DataFrame(data)
            self.assertTrue(df._mlir_builder is not None)
            self.assertTrue((df_orig + constant).equals((df + constant)._mock))
            self.assertTrue((df_orig - constant).equals((df - constant)._mock))
            self.assertTrue((df_orig * constant).equals((df * constant)._mock))
            self.assertTrue((df_orig / constant).equals((df / constant)._mock))
            self.assertTrue((df_orig % constant).equals((df % constant)._mock))

    def test_series(self) -> None:
        for name, data, constant in self.data_s:
            s: Series = Series(data, name=name)
            s_orig: Series = Series(data, name=name)
            self.assertTrue((s_orig + constant).equals((s + constant)._mock))
            self.assertTrue((s_orig - constant).equals((s - constant)._mock))
            self.assertTrue((s_orig * constant).equals((s * constant)._mock))
            self.assertTrue((s_orig / constant).equals((s / constant)._mock))
            self.assertTrue((s_orig % constant).equals((s % constant)._mock))
