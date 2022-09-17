"""Test handling of attributes not overwritten by DataFrame.

If an attribute/property/method would return a pd.DataFrame object,
it needs to be converted to a mlir_pandas.DataFrame object again.

Verify for each test that the tested attribute is present in
pd.DataFrame, but *not* in mlir_pandas.DataFrame.
"""

from typing import Any, Dict, List, Set, Union
import unittest

import pandas as pd

from mlir_pandas import DataFrame


class UnimplementedAttrTest(unittest.TestCase):
    data: List[Union[List[Any], Dict[str, Any]]] = [
        {"col1": [1, 2, 3], "col2": [4, 5, 6]},
        [1, 2, 3]
    ]
    real_data_only_attrs: Set[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.real_data_only_attrs = set().union(
            *(set(dir(pd.DataFrame(data))) for data in cls.data)
        ).difference(set().union(
            *(set(dir(DataFrame(data))) for data in cls.data)
        ))

    # TODO: attribute_returning_data?
    def test_attribute_not_returning_data(self) -> None:
        self.assertTrue("_typ" in self.real_data_only_attrs)
        self.assertTrue(pd.DataFrame()._typ == DataFrame()._typ)

    # TODO: property_returning_data?
    def test_property_not_returning_data(self) -> None:
        self.assertTrue("dtypes" in self.real_data_only_attrs)
        for data in self.data:
            self.assertTrue(pd.DataFrame(data).dtypes.equals(DataFrame(data).dtypes._mock))

    def test_method_not_returing_data(self) -> None:
        self.assertTrue("to_string" in self.real_data_only_attrs)
        for data in self.data:
            self.assertTrue(pd.DataFrame(data).to_string() == DataFrame(data).to_string())

    def test_method_returing_data(self) -> None:
        self.assertTrue("pivot" in self.real_data_only_attrs)
        # Use the example from the pandas documentation.
        data: Dict[str, Any] = {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"]
        }
        df: DataFrame = DataFrame(data).pivot(index="foo", columns="bar", values="baz")
        self.assertTrue(isinstance(df, DataFrame))
        self.assertFalse(isinstance(df, pd.DataFrame))
        self.assertTrue(
            pd.DataFrame(data).pivot(index="foo", columns="bar", values="baz").equals(df._mock)
        )
