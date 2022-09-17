# type: ignore

import unittest

import pandas as pd

from mlir_pandas import DataFrame as MLIRDataFrame


class SubclassingTest(unittest.TestCase):
    def test_subclassing(self) -> None:
        class DataFrame(MLIRDataFrame):
            pass
        class MyDataFrame(MLIRDataFrame):
            pass
        self.assertTrue(issubclass(MLIRDataFrame, MLIRDataFrame))
        for cl in [DataFrame, MyDataFrame]:
            self.assertFalse(issubclass(cl, MLIRDataFrame))
            self.assertTrue(issubclass(cl, pd.DataFrame))

    def test_duplicate_base_classes(self) -> None:
        with self.assertRaises(TypeError) as cm:
            class DataFrame(MLIRDataFrame, MLIRDataFrame):
                pass
        self.assertTrue(str(cm.exception) == "duplicate base class DataFrame")
