"""Map types to their mlir and pyarrow representation.

A mapping of supported numpy types and their representation in the
current MLIR dialect and the pyarrow type system.
The MLIR types has to be converted to strings if used as dictionary keys
as they are not hashable and thus cannot be directly used as dictionary
keys.
"""

import datetime
from typing import Any, Callable, Dict, Final, NamedTuple, Optional, Sequence, Tuple, Type

import mlir.astnodes as ma
import pandas as pd
import pyarrow as pa
from numpy import dtype
#from pandas._libs.arrays import NDArrayBacked
#from pandas._typing import Dtype, DtypeObj
from pandas._typing import DtypeObj
#from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
#from pandas.core.arrays import PandasArray
#from pandas.core.construction import extract_array
#from pandas.core.dtypes.common import pandas_dtype

from ._dialects.astnodes import DBNullable, DBStringType, DBTimeStamp, TimeUnit
#from ._util import Decimal


class SupportedTypesException(Exception):
    pass


# See https://github.com/pandas-dev/pandas/blob/v1.4.3/pandas/core/arrays/string_.py
# for an example on how to define an ExtensionDtype and an ExtensionArray.

#@register_extension_dtype
#class DecimalDType(ExtensionDtype):
#    _metadata: tuple[str, ...] = ("precision", "scale")
#
#    def __init__(self, precision: int, scale: int) -> None:
#        class SpecializedDecimal(Decimal):
#            precision: int = precision
#            scale: int = scale
#
#        self.precision: int = precision
#        self.scale: int = scale
#        self.decimal_class: Type[Decimal] = SpecializedDecimal
#
#    @classmethod
#    def construct_array_type(cls) -> Type[ExtensionArray]:
#        return DecimalArray
#
#    @property
#    def name(self) -> str:
#        return f"decimal({self.precision}, {self.scale})"
#
#    @property
#    def type(self) -> Type[Decimal]:
#        return self.decimal_class
#
#
#class DecimalArray(ExtensionArray, PandasArray):
#    def __init__(self, decimal_class: DecimalDType, values: Any, copy: bool = False) -> None:
#        self._decimal_class: DecimalDType = decimal_class
#        # Extract the array if "values" is a Series, for example.
#        values = extract_array(values)
#        super().__init__(values, copy=copy)
#        NDArrayBacked.__init__(self, self._ndarray, self._decimal_class)
#
#    @classmethod
#    def _from_sequence(cls, scalars: Any, *, dtype: Optional[Dtype] = None, copy: bool = False):
#        ...


class TypeInfoDType(NamedTuple):
    mlir: ma.Type
    pyarrow: pa.DataType
    python_type: Type[Any]


class TypeInfoMLIR(NamedTuple):
    dtype: DtypeObj
    pyarrow: pa.DataType
    python_type: Type[Any]


class TypeInfoPyArrow(NamedTuple):
    dtype: DtypeObj
    mlir: ma.Type
    python_type: Type[Any]


class TypeInfoPython(NamedTuple):
    dtype: DtypeObj
    mlir: ma.Type
    pyarrow: pa.DataType


def _map_to_int(n: int) -> Optional[TypeInfoPython]:
    if n < 2**64:
        return TypeInfoPython(dtype("int64"), DBNullable(ma.SignlessIntegerType(64)), pa.int64())
    return None


def create_nullable(type: ma.Type) -> DBNullable:
    return type if isinstance(type, DBNullable) else DBNullable(type)


def get_dtype_from_pyarrow_type(typ: pa.DataType) -> pa.DataType:
    if typ == pa.date32() or typ == pa.date64() or typ == pa.timestamp("ns"):
        return dtype("datetime64[ns]")
    try:
        return PyArrow2Pandas[typ]
    except KeyError:
        raise SupportedTypesException(f"{typ} is currently not supported")


def get_mlir_type_from_python_obj(obj: Any) -> ma.Type:
    # Get the MLIR type from the given Python object.
    try:
        type_info_python: Optional[TypeInfoPython] = TYPES_MAPPING_Python[type(obj)](obj)
    except KeyError:
        raise SupportedTypesException(f"{type(obj)} is currently not supported")
    if type_info_python is None:
        raise SupportedTypesException(f"could not find mlir type for {type(obj)}")
    return type_info_python.mlir


def get_mlir_value_from_python_obj(obj: Any) -> str:
    # Get the MLIR value from the given Python object.
    try:
        mlir_value: str = VALUE_MAPPING_Python[type(obj)](obj)
    except KeyError:
        raise SupportedTypesException(f"{type(obj)} is currently not supported")
    return mlir_value


def get_type(type: ma.Type) -> ma.Type:
    return type if not isinstance(type, DBNullable) else type.real_type


def pyarrow2pandas_types_mapper(t: pa.DataType) -> Optional[DtypeObj]:
    #if isinstance(t, pa.FixedSizeBinaryType):
    #    return pd.StringDtype()
    return PyArrow2Pandas.get(t, None)


_TYPES_LUT: Final[Sequence[Tuple[DtypeObj, ma.Type, pa.DataType, Type[Any]]]] = [
    (dtype("bool"), DBNullable(ma.SignlessIntegerType(1)), pa.bool_(), bool),
    (dtype("int64"), DBNullable(ma.SignlessIntegerType(64)), pa.int64(), int),
    (pd.Int64Dtype(), DBNullable(ma.SignlessIntegerType(64)), pa.int64(), int),
    (dtype("int32"), DBNullable(ma.SignlessIntegerType(32)), pa.int32(), int),
    (dtype("float64"), DBNullable(ma.FloatType(ma.FloatTypeEnum.f64)), pa.float64(), float),
    (pd.Float64Dtype(), DBNullable(ma.FloatType(ma.FloatTypeEnum.f64)), pa.float64(), float),
    (dtype("datetime64[s]"), DBNullable(DBTimeStamp(TimeUnit.second)), pa.timestamp("s"), datetime.datetime),
    (dtype("datetime64[ms]"), DBNullable(DBTimeStamp(TimeUnit.millisecond)), pa.timestamp("ms"), datetime.datetime),
    (dtype("datetime64[us]"), DBNullable(DBTimeStamp(TimeUnit.microsecond)), pa.timestamp("us"), datetime.datetime),
    (dtype("datetime64[ns]"), DBNullable(DBTimeStamp(TimeUnit.nanosecond)), pa.timestamp("ns"), datetime.datetime),
    (pd.StringDtype(), DBNullable(DBStringType()), pa.string(), str),
]
VALUE_MAPPING_Python: Final[Dict[Type[Any], Callable[[Any], str]]] = {
    bool: lambda b: "1" if b else "0",
    int: lambda i: str(i),
    float: lambda f: str(f),
    str: lambda s: f'"{s}"',
    datetime.datetime: lambda d: f'"{d.isoformat()}"'
}
TYPES_MAPPING_Python: Final[Dict[Type[Any], Callable[[Any], Optional[TypeInfoPython]]]] = {
    bool: lambda _: TypeInfoPython(dtype("bool"), DBNullable(ma.SignlessIntegerType(1)), pa.bool_()),
    int: _map_to_int,
    float: lambda _: TypeInfoPython(dtype("float64"), DBNullable(ma.FloatType(ma.FloatTypeEnum.f64)), pa.float64()),
    str: lambda _: TypeInfoPython(pd.StringDtype(), DBNullable(DBStringType()), pa.string()),
    # TODO: always nanosecond?
    datetime.datetime: lambda _: TypeInfoPython(dtype("datetime64[ns]"), DBNullable(DBTimeStamp(TimeUnit.nanosecond)), pa.timestamp("ns"))
}

TYPES_MAPPING_DType: Final[Dict[DtypeObj, TypeInfoDType]] = {
    dtype: TypeInfoDType(mlir, pyarrow, python_type)
    for dtype, mlir, pyarrow, python_type in _TYPES_LUT
}

TYPES_MAPPING_MLIR: Final[Dict[str, TypeInfoMLIR]] = {
    str(mlir): TypeInfoMLIR(dtype, pyarrow, python_type)
    for dtype, mlir, pyarrow, python_type in _TYPES_LUT
}

TYPES_MAPPING_PyArrow: Final[Dict[DtypeObj, TypeInfoPyArrow]] = {
    pyarrow: TypeInfoPyArrow(dtype, mlir, python_type)
    for dtype, mlir, pyarrow, python_type in _TYPES_LUT
}

# cf. https://arrow.apache.org/docs/python/pandas.html#nullable-types
PyArrow2Pandas: Dict[pa.DataType, DtypeObj] = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.float32(): pd.Float32Dtype(),
    pa.float64(): pd.Float64Dtype(),
    pa.string(): pd.StringDtype()
}


# cf. https://arrow.apache.org/docs/python/pandas.html#reducing-memory-use-in-table-to-pandas
PyArrow2PandasConversionArgs: Dict[str, Any] = {
    "date_as_object": False,
    "types_mapper": pyarrow2pandas_types_mapper,
    "split_blocks": True,
    "self_destruct": True
}

PyArrowFromPandasConversionArgs: Dict[str, Any] = {
    "preserve_index": False
}
