from typing import List, Optional, Union

import mlir.astnodes as ma
from mlir.builder.builder import DialectBuilder

from .astnodes import FunctionTypeAttr, GenericModule, GenericOperation, QuotedStr, StringAttr
from .common import ReturnValue


class BaseDialectException(Exception):
    pass


class MLIRDialectBuilder(DialectBuilder):
    """Implement base dialect features."""

    def call(
        self,
        func_name: str,
        operands: List[ReturnValue] = [],
        return_types: List[ma.Type] = []
    ) -> Union[ReturnValue, List[ReturnValue], None]:
        name: QuotedStr = QuotedStr("func.call")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[op.sym for op in operands],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("callee", ma.SymbolRefAttr([ma.SymbolRefId(func_name)]))
            ]),
            type=[ma.FunctionType([op.type for op in operands], return_types)]
        )

        results: Union[ma.SsaId, List[ma.SsaId], None]
        results = self.core_builder._insert_op_in_block([None] * len(return_types), op)

        if results is None:
            return None
        if not isinstance(results, list):
            return ReturnValue(results, return_types[0])

        if len(results) != len(return_types):
            raise BaseDialectException("unexpected results length")

        return [
            ReturnValue(result, return_type)
            for result, return_type in zip(results, return_types)
        ]

    def func(
        self,
        name: str,
        region: Optional[ma.Region] = None,
        operands: List[ReturnValue] = [],
        return_types: List[ma.Type] = [],
        insert: bool = False
    ) -> GenericOperation:
        operand_names: List[ma.SsaId] = [op.sym for op in operands]
        operand_types: List[ma.Type] = [op.type for op in operands]

        if region is None:
            region = ma.Region([
                ma.Block(ma.BlockLabel(ma.BlockId("bb0"), operand_names, operand_types), [])
                if operands else ma.Block(None, [])
            ])

        func: GenericOperation = GenericOperation(
            name=QuotedStr("func.func"),
            args=[],
            successors=None,
            regions=[region],
            attributes=ma.AttributeDict([
                ma.AttributeEntry("sym_name", StringAttr(name)),
                ma.AttributeEntry("function_type", FunctionTypeAttr(
                    ma.FunctionType(operand_types, return_types)
                ))
            ]),
            type=[ma.FunctionType([], [])]
        )

        if insert:
            self.core_builder._insert_op_in_block([], func)

        return func

    def module(self, region: Optional[ma.Region] = None) -> GenericModule:
        if region is None:
            region = ma.Region([ma.Block(None, [])])

        return GenericModule(
            name=QuotedStr("builtin.module"),
            args=[],
            region=region,
            attributes=None,
            type=ma.FunctionType([], [])
        )

    def return_op(self, results: List[ReturnValue]) -> None:
        self.core_builder._insert_op_in_block(
            [],
            GenericOperation(
                name=QuotedStr("func.return"),
                args=[res.sym for res in results],
                successors=None,
                regions=None,
                attributes=None,
                type=[ma.FunctionType([res.type for res in results], [])]
            ))
