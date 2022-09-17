from typing import Any, Optional

import mlir.astnodes as ma
from mlir_pandas._dialects.astnodes import DBNullable
from pandas._typing import DtypeObj

from mlir_pandas._supported_types import TypeInfoMLIR, TYPES_MAPPING_DType, TYPES_MAPPING_MLIR
from mlir_pandas._util import OrderedSet


class ReturnValue:
    def __init__(self, ret_sym: Any, ret_type: ma.Type) -> None:
        if not isinstance(ret_sym, ma.SsaId):
            raise ValueError(f"sym type is {type(ret_sym)} instead of ma.SsaId")
        self.sym: ma.SsaId = ret_sym
        self.type: ma.Type = ret_type

    def __repr__(self) -> str:
        return f"ReturnValue({self.sym}, {self.type})"


class TableColumn:
    def __init__(self, scope: str, name: str, type: ma.Type) -> None:
        self.scope: str = scope
        self.name: str = name
        self.type: ma.Type = type
        self.other_types: TypeInfoMLIR = TYPES_MAPPING_MLIR[str(
            #type.real_type if isinstance(type, DBNullable) else type
            type if isinstance(type, DBNullable) else DBNullable(type)
        )]

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, type(self)):
            return False
        other: "TableColumn" = obj
        return self.scope == other.scope and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.scope + self.name)

    def __repr__(self) -> str:
        return f"({self.scope}, {self.name}, {self.type})"

    def to_symbol_ref_attr(self) -> ma.SymbolRefAttr:
        return ma.SymbolRefAttr([ma.SymbolRefId(self.scope), ma.SymbolRefId(self.name)])


class TableColumnSL:
    """A scopeless TableColumn."""

    def __init__(self, name: str, type: ma.Type) -> None:
        self.name: str = name
        self.type: ma.Type = type
        self.other_types: TypeInfoMLIR = TYPES_MAPPING_MLIR[str(
            type.real_type if isinstance(type, DBNullable) else type
        )]

    @classmethod
    def from_dtype(cls, name: str, type: DtypeObj) -> "TableColumnSL":
        # TODO: throw custom exception.
        return cls(name, TYPES_MAPPING_DType[type].mlir)

    def __repr__(self) -> str:
        return f"({self.name}, {self.type})"

    def col(self, scope: str, name: Optional[str] = None) -> TableColumn:
        return TableColumn(scope, self.name if name is None else name, self.type)


class TableContext:
    def __init__(self, id: str, cols: OrderedSet[TableColumn]) -> None:
        self.id: str = id
        self.cols: OrderedSet[TableColumn] = cols
