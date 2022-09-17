#!/usr/bin/env python3

import logging
import pandas as pd
import pyarrow as pa

from mlir_pandas import DataFrame

#import logging
#logging.basicConfig(level=logging.DEBUG)

df = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
df2 = DataFrame({"col3": [5, 4, 6], "col4": [7, 8, 9]})

print(df.merge(df2, how="inner", left_on=["col2"], right_on=["col3"]))
