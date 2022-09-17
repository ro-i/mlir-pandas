import logging
import os
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from multiprocessing import Queue, Process
from resource import RUSAGE_SELF, RUSAGE_CHILDREN, getrusage
from typing import Any, Callable, Dict, List, Literal, Match, Optional, Tuple, Union

import modin.pandas as modin_pd
import pandas as pd
import pyarrow as pa

# Modin-related. ray does not seem to consider for its memory consumption
# computations that /tmp can be a tmpfs. So temp_dir should be not in memory.
#import ray
#ray.init(include_dashboard=False, _temp_dir="/home/robert/tmp_ray")

sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
from pymlirdbext import load as db_load, run as db_run

from redirect_stdout import stdout_redirector # type: ignore

sys.path.append("..")
from mlir_pandas.mlir_pandas import DataFrame
from mlir_pandas._supported_types import PyArrow2PandasConversionArgs
from mlir_pandas._util import gettime


DF = Union[modin_pd.DataFrame, pd.DataFrame, DataFrame]


@dataclass
class Duration:
    ns: float

    def __repr__(self) -> str:
        us: float = self.ns / 1e3
        ms: float = self.ns / 1e6
        s: float = self.ns / 1e9
        if s > 1:
            return f"{round(s, 2)} s"
        elif ms > 1:
            return f"{round(ms, 2)} ms"
        elif us > 1:
            return f"{round(us)} us"
        else:
            return f"{round(self.ns)} ns"


@dataclass
class MemUsage:
    kb: float

    def __repr__(self) -> str:
        mb: float = self.kb / 1e3
        gb: float = self.kb / 1e6
        if gb > 1:
            return f"{round(gb, 2)} GB"
        elif mb > 1:
            return f"{round(mb, 2)} MB"
        else:
            return f"{round(self.kb)} KB"


class BenchmarkResult:
    def __init__(
        self,
        result: Any,
        duration: float,
        mem_usage: float,
        timestamps: Optional[List[Tuple[str, str, Duration]]],
        db_times: Optional[Dict[str, float]]
    ) -> None:
        self.result: Any = result
        self.duration: Duration = Duration(duration)
        self.mem_usage: MemUsage = MemUsage(mem_usage)
        self.timestamps: Optional[List[Tuple[str, str, Duration]]] = timestamps
        self.db_times: Optional[Dict[str, Duration]] = None

        if db_times is not None:
            self.db_times = {n: Duration(d) for n, d in db_times.items()}

    def __repr__(self) -> str:
        ts_durations: Optional[str] = None
        if self.timestamps is not None:
            ts_durations = "\n".join(
                f"between {n1} and {n2}: {d}" for n1, n2, d in self.timestamps[:-1]
            )
            n1, n2, d = self.timestamps[-1]
            ts_durations += f"\ntotal time bewteen {n1} and {n2}: {d}"

        return (
            (f"> result:\n{self.result}"
             f"\n> total duration: {self.duration}"
             f"\n> maximum memory usage: {self.mem_usage}")
            + (f"\n> timestamps:\n{ts_durations}" if ts_durations is not None else "")
            + (f"\n> db_times:\n{self.db_times}" if self.db_times is not None else "")
        )


@dataclass
class TPCHResult:
    name: str
    pandas: BenchmarkResult
    mlir_pandas: BenchmarkResult
    mlir: BenchmarkResult
    #modin: BenchmarkResult

    def __repr__(self) -> str:
        return (f">>>>> results for {self.name}"
                f"\n>>- pandas:\n{self.pandas}"
                f"\n>>- mlir_pandas:\n{self.mlir_pandas}"
                f"\n>>- mlir:\n{self.mlir}")
                #f"\n>>- modin:\n{self.modin}")


def execute_mlir(
    tables: List[pa.Table],
    table_identifiers: List[str],
    mlir_module: str
) -> Tuple[List[Tuple[str, int]], pd.DataFrame]:
    timestamps: List[Tuple[str, int]] = []

    timestamps.append(("pre load to db", gettime()))
    db_load({
        table_identifier: table
        for table_identifier, table in zip(table_identifiers, tables)
    })
    timestamps.append(("post load to db", gettime()))

    timestamps.append(("pre run", gettime()))
    res: pa.Table = db_run(mlir_module)
    timestamps.append(("post run", gettime()))

    timestamps.append(("pre load from db", gettime()))
    ret: pd.DataFrame = res.to_pandas(**PyArrow2PandasConversionArgs)
    del res
    timestamps.append(("post load from db", gettime()))

    return (timestamps, ret)


def execute_modin(
    func: Callable[..., Any],
    modin_dfs: List[modin_pd.DataFrame]
) -> Tuple[List[Tuple[str, int]], Any]:
    timestamps: List[Tuple[str, int]] = []

    timestamps.append(("pre run", gettime()))
    ret: Any = func(*modin_dfs)
    timestamps.append(("post run", gettime()))

    return (timestamps, ret)


def make_mlir_pandas(table: pa.Table) -> DataFrame:
    ret: DataFrame = DataFrame.from_pyarrow_table(table)
    # Verify that the data is supported.
    assert(ret._mlir_builder is not None)
    return ret


def make_mlir_pandas_from_df(df: pd.DataFrame) -> DataFrame:
    ret: DataFrame = DataFrame(df)
    # Verify that the data is supported.
    assert(ret._mlir_builder is not None)
    return ret


def measure_time(
    func: Callable[..., Any],
    benchmark_type: Literal["pandas", "mlir_pandas", "mlir", "modin"],
    *args: Any,
    **kwargs: Any
) -> BenchmarkResult:
    def execute_func(queue: "Queue[Any]", func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        start: int = gettime()
        result_func: Any = func(*args, **kwargs)
        end: int = gettime()
        # maxrss is in KB
        maxmem: int = max(getrusage(RUSAGE_SELF).ru_maxrss, getrusage(RUSAGE_CHILDREN).ru_maxrss)
        queue.put_nowait((end - start, maxmem, result_func))

    def timestamp_add(p1: str, t1: int, p2: str, t2: int) -> Tuple[str, int]:
        assert(p1 == p2)
        return (p1, t1 + t2)

    iterations: int = 10

    duration: int = 0
    mem_usage: float = 0
    timestamps: List[Tuple[str, int]] = []
    db_times: Dict[str, Tuple[float, int]] = {name: (0, 0) for name, _ in DB_OUTPUT_PATTERNS}

    result: Any
    result_timestamps: List[Tuple[str, int]] = []
    queue: "Queue[Any]" = Queue(maxsize=1)

    stdout: BytesIO = BytesIO()
    with stdout_redirector(stdout):
        for _ in range(iterations):
            p: Process = Process(target=execute_func, args=(queue, func) + args, kwargs=kwargs)
            p.start()
            p.join()

            d, mem, result_func = queue.get_nowait()

            duration += d
            mem_usage += mem

            if benchmark_type in ["mlir_pandas", "mlir", "modin"]:
                result_timestamps, result = result_func

                if not timestamps:
                    timestamps = result_timestamps
                else:
                    timestamps = [
                        timestamp_add(p1, t1, p2, t2)
                        for (p1, t1), (p2, t2) in zip(timestamps, result_timestamps)
                    ]
            else:
                result = result_func
    output: str = stdout.getvalue().decode("utf-8")

    if benchmark_type == "mlir_pandas" or benchmark_type == "mlir":
        for line in output.splitlines():
            for name, pattern in DB_OUTPUT_PATTERNS:
                m: Optional[Match[str]] = re.fullmatch(pattern, line)
                if m is None:
                    continue
                result_ns: float = float(m.group(1)) * 1e6
                res, count = db_times[name]
                db_times[name] = (res + result_ns, count + 1)

    logging.debug(output)

    duration_result: float = duration / iterations
    mem_usage_result: float = mem_usage / iterations
    timestamp_diffs: Optional[List[Tuple[str, str, Duration]]] = None
    if timestamps:
        timestamp_diffs = []
        ts: Tuple[str, int] = timestamps[0]
        for i in range(1, len(timestamps)):
            ts2: Tuple[str, int] = timestamps[i]
            timestamp_diffs.append((
                ts[0], ts2[0], Duration((ts2[1] - ts[1]) / iterations)
            ))
            ts = ts2
        timestamp_diffs.append((
            timestamps[0][0], timestamps[-1][0],
            Duration((timestamps[-1][1] - timestamps[0][1]) / iterations)
        ))

    assert(c == iterations for _, c in db_times.values())
    db_times_result: Dict[str, float] = {
        n: r / iterations for n, (r, _) in db_times.items()
    }

    return BenchmarkResult(
        result, # type: ignore
        duration_result,
        mem_usage_result,
        timestamp_diffs,
        db_times_result if benchmark_type == "mlir_pandas" or benchmark_type == "mlir" else None
    )


def run(
    benchmark_name: str,
    db_files: List[str],
    alt_schemata: List[List[Tuple[str, pa.DataType]]],
    mlir_func: Callable[..., Any],
    mlir_pandas_func: Callable[..., Any],
    pandas_func: Callable[..., Any],
    modin_func: Callable[..., Any]
) -> TPCHResult:
    tables: List[pa.Table] = []
    # cf. https://arrow.apache.org/docs/python/ipc.html#efficiently-writing-and-reading-arrow-data
    for db_file in db_files:
        with pa.OSFile(db_file, "rb") as source:
            tables.append(pa.ipc.open_file(source).read_all())

    result_mlir: BenchmarkResult = measure_time(mlir_func, "mlir", *tables)

    # Convert to alternative schema for mlir_pandas.
    tables = [table.cast(pa.schema(alt_schema)) for table, alt_schema in zip(tables, alt_schemata)]
    result: BenchmarkResult = measure_time(mlir_pandas_func, "mlir_pandas", *tables)

    # Convert from the alternative schema to a pd.DataFrame.
    df_tables: List[DF] = []
    for table in tables:
        df_tables.append(table_to_df_and_invalidate_table(table))
        del table
    del tables

    result_orig: BenchmarkResult = measure_time(pandas_func, "pandas", *df_tables)

    # Convert from the pd.DataFrame to a modin DataFrame.
    #df_tables = [pandas_to_modin(df_table) for df_table in df_tables]
    #result_modin: BenchmarkResult = measure_time(modin_func, "modin", *df_tables)

    return TPCHResult(benchmark_name, result_orig, result, result_mlir)


def pandas_to_modin(df: pd.DataFrame) -> modin_pd.DataFrame:
    return modin_pd.DataFrame(df)


def table_to_df_and_invalidate_table(table: pa.Table) -> pd.DataFrame:
    return table.to_pandas(**PyArrow2PandasConversionArgs)


DB_OUTPUT_PATTERNS: List[Tuple[str, str]] = [
    ("optimization", r"optimization took: ([0-9\.])+ ms"),
    ("lowering to db", r"lowering to db took: ([0-9\.])+ ms"),
    ("lowering to std", r"lowering to std took: ([0-9\.])+ ms"),
    ("lowering to llvm", r"lowering to llvm took: ([0-9\.])+ ms"),
    ("conversion", r"conversion: ([0-9\.])+ ms"),
    ("jit", r"jit: ([0-9\.])+ ms"),
    ("runtime", r"runtime: ([0-9\.])+ ms")
]
