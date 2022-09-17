from typing import Any, Dict
import unittest

import pandas as pd

from mlir_pandas import DataFrame


class JoinTest(unittest.TestCase):
    data1: Dict[str, Any] = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    data2: Dict[str, Any] = {"col3": [1, 2, 3], "col4": [7, 8, 9]}

    def test_join(self) -> None:
        df1: DataFrame = DataFrame(self.data1)
        df1_orig: pd.DataFrame = pd.DataFrame(self.data1)
        df2: DataFrame = DataFrame(self.data2)
        df2_orig: pd.DataFrame = pd.DataFrame(self.data2)

        df3: DataFrame = df1.merge(df2, how="inner", left_on="col1", right_on="col3")
        df3_orig: pd.DataFrame = df1_orig.merge(
            df2_orig, how="inner", left_on="col1", right_on="col3"
        )
        df3_orig.index = df3_orig.index.astype("Int64")
        self.assertTrue(df3_orig.astype("Int64").equals(df3._mock))
