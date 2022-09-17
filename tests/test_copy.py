from typing import Any, Dict, List, Union
import unittest

from mlir_pandas import DataFrame


# Note:
# In contrast to "==", which checks for equality, "is" checks if the
# given objects are *identical*.


class CopyTest(unittest.TestCase):
    supported_data: List[Union[List[Any], Dict[str, Any]]] = [
        {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    ]
    unsupported_data: List[Union[List[Any], Dict[str, Any]]] = [
        [lambda x: x + 1]
    ]

    def test_copy_empty(self) -> None:
        df1: DataFrame = DataFrame()
        df2: DataFrame = df1.copy()
        self.assertTrue(df1._mlir_builder is not df2._mlir_builder)

    def test_copy_supported(self) -> None:
        for data in self.supported_data:
            df1: DataFrame = DataFrame(data)
            df2: DataFrame = df1.copy()
            self.assertTrue(df1._mlir_builder is not df2._mlir_builder)

    def test_copy_unsupported(self) -> None:
        for data in self.unsupported_data:
            df1: DataFrame = DataFrame(data)
            df2: DataFrame = df1.copy()
            self.assertTrue(df1._mlir_builder is df2._mlir_builder is None)
