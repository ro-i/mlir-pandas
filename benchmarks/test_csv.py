#!/usr/bin/python3

import sys
from typing import Any, List, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

sys.path.append("..")
from mlir_pandas.mlir_pandas import DataFrame

import common


DF = Union[pd.DataFrame, DataFrame]


def test_csv(df_table: DF) -> Any:
    return df_table[
        (df_table["col1"] < df_table["col2"]) & (df_table["col2"] < df_table["col3"])
    ]


def test_csv_mlir_pandas(table: pa.Table) -> Tuple[List[Tuple[str, int]], Any]:
    df_table: DataFrame = common.make_mlir_pandas(table)
    first_ts: int = df_table._timestamps[0][1]
    result: Any = test_csv(df_table)
    assert(result._timestamps[0][1] == first_ts)
    result_materialized: Any = result._mock
    return (result._timestamps, result_materialized)


def run(csv_size: int) -> None:
    table: pa.Table = pa_csv.read_csv(
        f"../rand_{csv_size}.csv",
        read_options=pa_csv.ReadOptions(column_names=["col1", "col2", "col3"])
    )
    table = table.cast(pa.schema([
        ("col1", pa.int64()),
        ("col2", pa.int64()),
        ("col3", pa.int64()),
    ]))

    result: common.BenchmarkResult = common.measure_time(
        test_csv_mlir_pandas, "mlir_pandas", table
    )
    print(result)

    #df_table = common.table_to_df_and_invalidate_table(table)
    #del table
    #result_orig: common.BenchmarkResult = common.measure_time(
    #    test_csv, "pandas", df_table
    #)

    #print("Test CSV\n", result_orig, "\n", result)


if __name__ == "__main__":
    run(int(1e6))
