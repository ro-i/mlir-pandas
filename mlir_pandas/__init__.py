from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame, Series
    from pandas.core.groupby import DataFrameGroupBy
else:
    from .mlir_pandas import DataFrame, DataFrameGroupBy, Series

__all__ = ["DataFrame", "DataFrameGroupBy", "Series"]
