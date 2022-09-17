from typing import Any, Dict, cast
import unittest

import pandas as pd

from mlir_pandas import DataFrame


class RenameTest(unittest.TestCase):
    data: Dict[str, Any] = {"col1": [1, 2, 3], "col2": [4, 5, 6]}

    def test_rename(self) -> None:
        df: DataFrame = cast(DataFrame, DataFrame(self.data).rename(columns={"col1": "col3"}))
        df_orig: pd.DataFrame = cast(pd.DataFrame, pd.DataFrame(self.data).rename(
            columns={"col1": "col3"}
        ))
        self.assertTrue(df_orig.equals(df._mock))
