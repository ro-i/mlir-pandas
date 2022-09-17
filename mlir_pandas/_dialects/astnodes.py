from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Literal, Optional, Tuple

import mlir.astnodes as ma


"""Additional ast nodes."""


@dataclass
class ColumnDefAttr(ma.Attribute):
    value: Tuple[ma.SymbolRefAttr, ma.Type]

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("#relalg.columndef<{},{}>".format(
            *map(lambda v: ma.dump_or_value(v, indent), self.value)
        ))


@dataclass
class ColumnRenameDefAttr(ma.Attribute):
    new: ma.SymbolRefAttr
    type: ma.Type
    old: "ColumnRefAttr"

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("#relalg.columndef<{},{},[{}]>".format(
            *map(lambda v: ma.dump_or_value(v, indent), (self.new, self.type, self.old))
        ))


@dataclass
class ColumnRefAttr(ma.Attribute):
    value: ma.SymbolRefAttr

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("#relalg.columnref<{}>".format(
            ma.dump_or_value(self.value, indent)
        ))


@dataclass
class DBCmpPredicateAttr(ma.IntegerAttr):
    value: Any
    type: ma.Type = ma.SignlessIntegerType(64)


@dataclass
class DBDecimal(ma.Type):
    value: int
    precision: int

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("!db.decimal<{},{}>".format(self.value, self.precision), indent)


@dataclass
class DBNullable(ma.Type):
    real_type: ma.Type

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("!db.nullable<{}>".format(self.real_type.dump(indent)), indent)


@dataclass
class DBStringType(ma.Type):
    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("!db.string", indent)


@dataclass
class DBTableType(ma.Type):
    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("!dsa.table", indent)


@dataclass
class DBTimeStamp(ma.Type):
    precision: "TimeUnit"

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("!db.timestamp<{}>".format(self.precision.name), indent)


@dataclass
class FloatType(ma.FloatType):
    type: "FloatTypeEnum"


@dataclass
class FloatTypeEnum(Enum):
    float32 = "float32"
    float64 = "float64"


@dataclass
class FunctionTypeAttr(ma.Attribute):
    value: ma.FunctionType

    def dump(self, indent: int = 0) -> str:
        return self.value.dump(indent)


@dataclass
class GenericModule(ma.GenericModule):
    # Fix wrong typing for "type" attribute.
    type: ma.FunctionType


@dataclass
class GenericOperation(ma.GenericOperation):
    # Add a regions attribute.
    # cf. https://mlir.llvm.org/docs/LangRef/#operations
    regions: Optional[List[ma.Region]]

    def dump(self, indent: int = 0) -> str:
        """Adapted copy of original implementation."""
        result = '%s' % self.name
        result += '('

        if self.args:
            result += ', '.join(ma.dump_or_value(arg, indent) for arg in self.args)

        result += ')'
        if self.successors:
            result += '[' + ma.dump_or_value(self.successors, indent) + ']'
        if self.regions:
            result += ' ( ' + ', '.join(r.dump(indent) for r in self.regions) + ')'
        if self.attributes:
            result += ' ' + ma.dump_or_value(self.attributes, indent)
        if isinstance(self.type, list):
            result += ' : ' + ', '.join(
                ma.dump_or_value(t, indent) for t in self.type)
        else:
            result += ' : ' + ma.dump_or_value(self.type, indent)
        return result


@dataclass
class QuotedStr(ma.Type, str):
    value: str

    def __str__(self) -> str:
        return '"{}"'.format(super().__str__())

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value('"{}"'.format(self.value), indent)


@dataclass
class RelalgAggrFuncAttr(ma.IntegerAttr):
    value: Any
    type: ma.Type = ma.SignlessIntegerType(64)


@dataclass
class RelalgSetSemanticAttr(ma.IntegerAttr):
    value: Literal[0, 1]
    type: ma.SignlessIntegerType = ma.SignlessIntegerType(64)

    def __post_init__(self) -> None:
        assert(self.value in [0, 1])
        assert(self.type.width == 64)


@dataclass
class RelalgSortSpecAttr(ma.Attribute):
    attr: ma.SymbolRefAttr
    sort_spec: Literal[0, 1]

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value(
            "#relalg.sortspec<{},{}>".format(
                self.attr.dump(indent),
                ma.dump_or_value("desc" if self.sort_spec == 0 else "asc", indent)
            ),
            indent
        )


@dataclass
class RelalgTupleStreamType(ma.Type):
    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("!relalg.tuplestream", indent)


@dataclass
class RelalgTupleType(ma.Type):
    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value("!relalg.tuple", indent)


@dataclass
class StringAttr(ma.StringAttr):
    type: Optional[ma.Type] = None

    def dump(self, indent: int = 0) -> str:
        return '"{}"'.format(super().dump(indent))


@dataclass
class TableMetaDataAttr(ma.Attribute):
    value: QuotedStr

    def dump(self, indent: int = 0) -> str:
        return ma.dump_or_value('#relalg.table_metadata<{}>'.format(
            ma.dump_or_value(self.value, indent)
        ))


@dataclass
class TimeUnit(Enum):
    second      = auto()
    millisecond = auto()
    microsecond = auto()
    nanosecond  = auto()
