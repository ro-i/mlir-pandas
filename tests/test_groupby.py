from typing import Any, Dict
import unittest

import numpy as np
import pandas as pd

from mlir_pandas import DataFrame


class GroupByTest(unittest.TestCase):
    data: Dict[str, Any] = {"col1": [1, 2, 1, 3], "col2": [5, 5, 6, 7], "col3": [7, 8, 9, 10]}

    def test_groupby(self) -> None:
        df: DataFrame = DataFrame(self.data)
        df_orig: pd.DataFrame = pd.DataFrame(self.data)

        # index=True (default) currently not supported
        self.assertTrue(
            df_orig.groupby("col1", as_index=False).aggregate(s=("col2", "sum")).astype("Int64").equals(
                df.groupby("col1", as_index=False).aggregate(s=("col2", "sum"))._mock
            )
        )

        self.assertTrue(
            df_orig.groupby("col1", as_index=False).aggregate(
                s=("col2", "sum"), m=("col2", "mean"), m2=("col3", "max")
            ).astype(np.int64).astype("Int64").equals(
                df.groupby("col1", as_index=False).aggregate(
                    s=("col2", "sum"), m=("col2", "mean"), m2=("col3", "max")
                )._mock
            )
        )
