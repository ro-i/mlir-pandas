import unittest
from datetime import datetime
from typing import Any, Dict, List, Union

import pandas as pd

from mlir_pandas import DataFrame, Series


class InitTest(unittest.TestCase):
    supported_data: List[Dict[str, Any]] = [
        {"col1": [1, 2, 3], "col2": [4, 5, 6]},
        {"col1": [1.5, 2, 3], "col2": [4, 5.3, 6]},
        {"col1": [True, False, True], "col2": [True, False, False]},
        {"col1": [datetime.now(), datetime.fromisoformat("2022-07-07")], "col2": [4, 5]}
    ]
    unsupported_data: List[Union[List[Any], Dict[str, Any]]] = [
        [1, 2, 3],
        [lambda x: x + 1],
        {"col1": [lambda x: x + 1]}
    ]

    def test_init_empty(self) -> None:
        self.assertTrue(pd.DataFrame().equals(DataFrame()._mock))

    def test_init_supported(self) -> None:
        for data in self.supported_data:
            df: DataFrame = DataFrame(data)
            self.assertTrue(df._mlir_builder is not None)
            self.assertTrue(pd.DataFrame(data).equals(df._mock))
            for n, d in data.items():
                s: Series = Series(d, name=n)
                self.assertTrue(s._mlir_builder is not None)
                self.assertTrue(pd.Series(d, name=n).equals(s._mock))

    def test_init_unsupported(self) -> None:
        for data in self.unsupported_data:
            df: DataFrame = DataFrame(data)
            self.assertTrue(df._mlir_builder is None)
            self.assertTrue(pd.DataFrame(data).equals(df._mock))
            s: Series
            if isinstance(data, dict):
                for n, d in data.items():
                    s = Series(d, name=n)
                    self.assertTrue(s._mlir_builder is None)
                    self.assertTrue(pd.Series(d, name=n).equals(s._mock))
            else:
                s = Series(data)
                self.assertTrue(pd.Series(data).equals(s._mock))
