import unittest
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from mlir_pandas import DataFrame, Series


class SlicingTest(unittest.TestCase):
    supported_data: List[Tuple[Union[List[Any], Dict[str, Any]], Union[List[Any], Any]]] = [
        ({"col1": [1, 2, 3], "col2": [4, 5, 6]}, "col1"),
        ({"col1": [1, 2, 3], "col2": [4, 5, 6]}, ["col2"]),
        ({"col1": [1, 2, 3], "col2": [4, 5, 6]}, ["col1", "col2"])
    ]
    unsupported_data: List[Tuple[Union[List[Any], Dict[str, Any]], Union[List[Any], Any]]] = [
        ([lambda x: x + 1], 0)
    ]

    def test_basic_slicing(self) -> None:
        for data, columns in self.supported_data + self.unsupported_data:
            slice1: Union[pd.Series, pd.DataFrame] = pd.DataFrame(data)[columns]
            slice2: Union[pd.Series, Series, DataFrame] = DataFrame(data)[columns]
            self.assertTrue(
                slice1.equals(slice2 if isinstance(slice2, pd.Series) else slice2._mock)
            )

    def test_unknown_columns(self) -> None:
        with self.assertRaises(KeyError) as cm:
            DataFrame({"col1": [1, 2, 3]})["col2"]
        self.assertTrue(str(cm.exception) == "'col2'")
