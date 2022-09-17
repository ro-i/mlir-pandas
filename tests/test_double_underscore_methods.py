from typing import Any, Dict, List, Optional, Union
import unittest

import pandas as pd

from mlir_pandas import DataFrame


class DoubleUnderscoreMethodsTest(unittest.TestCase):
    data: List[Optional[Union[List[Any], Dict[str, Any]]]] = [
        None,
        {"col1": [1, 2, 3], "col2": [4, 5, 6]},
        [1, 2, 3],
        [lambda x: x + 1]
    ]

    def test_str(self) -> None:
        for data in self.data:
            self.assertTrue(str(pd.DataFrame(data)) == str(DataFrame(data)))

    def test_repr(self) -> None:
        for data in self.data:
            self.assertTrue(str(pd.DataFrame(data).__repr__() == DataFrame(data).__repr__()))

    def test_subscriptable(self) -> None:
        data: List[int] = [1, 2, 3]
        self.assertTrue(pd.DataFrame(data)[0].equals(DataFrame(data)[0]._mock))
