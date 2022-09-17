from typing import Literal, Optional

import mlir.astnodes as ma
from mlir.builder.builder import DialectBuilder

from .._supported_types import create_nullable, get_type
from .astnodes import DBCmpPredicateAttr, DBNullable, GenericOperation, QuotedStr
from .common import ReturnValue


class DBDialectException(Exception):
    pass


class MLIRDialectBuilder(DialectBuilder):
    """Implement the db dialect."""

    # TODO: better handling of types for the arithmetic operations needed?
    # Are different types for left/right operand in some cases ok?

    def _arith_op(self, arith_op: str, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        if left.type != right.type:
            raise DBDialectException(f"{left.type} does not match {right.type}")

        name: QuotedStr = QuotedStr(arith_op)
        return_type: ma.Type = left.type

        op: GenericOperation = GenericOperation(
            name=name,
            args=[left.sym, right.sym],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType([left.type, right.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def _bool_op(
        self,
        bool_op: str,
        left: ReturnValue,
        right: Optional[ReturnValue] = None
    ) -> ReturnValue:
        if get_type(left.type) != ma.SignlessIntegerType(1):
            raise DBDialectException(f"{left.type} is not type bool (i1)")
        if right is not None and get_type(right.type) != ma.SignlessIntegerType(1):
            raise DBDialectException(f"{right.type} is not type bool (i1)")

        name: QuotedStr = QuotedStr(bool_op)
        return_type: ma.Type = left.type

        op: GenericOperation = GenericOperation(
            name=name,
            args=[left.sym, right.sym] if right is not None else [left.sym],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType(
                [left.type, right.type] if right is not None else [left.type],
                [return_type]
            )]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def add(self, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        return self._arith_op("db.add", left, right)

    def and_op(self, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        return self._bool_op("db.and", left, right)

    def as_nullable(self, operand: ReturnValue, null: bool = False) -> ReturnValue:
        # TODO: null currently not implemented
        name: QuotedStr = QuotedStr("db.as_nullable")
        return_type: ma.Type = DBNullable(operand.type)

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def cast(self, operand: ReturnValue, new_type: ma.Type) -> ReturnValue:
        name: QuotedStr = QuotedStr("db.cast")
        return_type: ma.Type = create_nullable(new_type)

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def compare(
        self,
        left: ReturnValue,
        right: ReturnValue,
        predicate: Literal[0, 1, 2, 3, 4, 5]
    ) -> ReturnValue:
        if predicate not in range(6):
            raise DBDialectException()

        name: QuotedStr = QuotedStr("db.compare")
        return_type: ma.Type = DBNullable(ma.SignlessIntegerType(1))
        #return_type: ma.Type = ma.SignlessIntegerType(1)

        op: GenericOperation = GenericOperation(
            name=name,
            args=[left.sym, right.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("predicate", DBCmpPredicateAttr(predicate))
            ]),
            type=[ma.FunctionType([left.type, right.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def constant(self, constant: ma.PrimitiveAttribute) -> ReturnValue:
        name: QuotedStr = QuotedStr("db.constant")
        return_type: ma.Type = constant.type

        op: GenericOperation = GenericOperation(
            name=name,
            args=[],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([ma.AttributeEntry("value", constant)]),
            type=[ma.FunctionType([], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def constant_true(self) -> ReturnValue:
        return self.constant(ma.PrimitiveAttribute(1, ma.SignlessIntegerType(1)))

    def div(self, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        return self._arith_op("db.div", left, right)

    def isnull(self, operand: ReturnValue) -> ReturnValue:
        name: QuotedStr = QuotedStr("db.isnull")
        return_type: ma.Type = ma.SignlessIntegerType(1)

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType([create_nullable(operand.type)], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def mod(self, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        return self._arith_op("db.mod", left, right)

    def mul(self, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        return self._arith_op("db.mul", left, right)

    def null(self, real_type: ma.Type) -> ReturnValue:
        name: QuotedStr = QuotedStr("db.null")
        return_type: ma.Type = create_nullable(real_type)

        op: GenericOperation = GenericOperation(
            name=name,
            args=[],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType([], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def not_op(self, left: ReturnValue, _: Optional[ReturnValue] = None) -> ReturnValue:
        return self._bool_op("db.not", left)

    def or_op(self, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        return self._bool_op("db.or", left, right)

    def sub(self, left: ReturnValue, right: ReturnValue) -> ReturnValue:
        return self._arith_op("db.sub", left, right)

