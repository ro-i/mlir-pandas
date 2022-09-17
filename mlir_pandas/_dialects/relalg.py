import json
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import mlir.astnodes as ma
from mlir.builder.builder import DialectBuilder

from mlir_pandas._util import OrderedDict, OrderedSet
from .astnodes import (
    DBNullable,
    RelalgAggrFuncAttr,
    ColumnDefAttr,
    ColumnRenameDefAttr,
    ColumnRefAttr,
    DBTableType,
    GenericOperation,
    QuotedStr,
    RelalgSetSemanticAttr,
    RelalgSortSpecAttr,
    RelalgTupleStreamType,
    RelalgTupleType,
    StringAttr,
    TableMetaDataAttr
)
from .common import ReturnValue, TableColumn, TableContext


class RelalgDialectException(Exception):
    pass


class RelalgOperationalError(Exception):
    pass


class MLIRDialectBuilder(DialectBuilder):
    """Implement the relalg dialect."""

    @staticmethod
    def _check_tuple_stream_like(type: ma.Type) -> bool:
        """Check whether the given type is like a tuple stream.

        Currently applies for DBTableType and RelalgTupleStreamType.
        """
        # TODO: all?
        return isinstance(type, DBTableType) or isinstance(type, RelalgTupleStreamType)

    def aggregation(
        self,
        table_context: TableContext,
        operand: ReturnValue,
        group_by_cols: List[TableColumn],
        computed_cols: List[TableColumn],
        region: ma.Region
    ) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.aggregation")
        return_type: ma.Type = RelalgTupleStreamType()

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=[region],
            attributes=ma.AttributeDict([
                ma.AttributeEntry("group_by_cols", ma.ArrayAttr([
                    ColumnRefAttr(tc.to_symbol_ref_attr()) for tc in group_by_cols
                ])),
                ma.AttributeEntry("computed_cols", ma.ArrayAttr([
                    ColumnDefAttr((tc.to_symbol_ref_attr(), tc.type)) for tc in computed_cols
                ]))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        table_context.cols = OrderedSet(group_by_cols + computed_cols)

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def aggrfn(
        self,
        operand: ReturnValue,
        aggrfn: Literal[0, 1, 2, 3, 4],
        column: TableColumn,
        #table_context: TableContext,
        #cast_to_float_region: ma.Region
    ) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.aggrfn")
        return_type: ma.Type = column.type
        #return_type: ma.Type = (
        #    DBNullable(ma.FloatType(ma.FloatTypeEnum.f64)) if aggrfn == 3 else column.type
        #)

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")
        if aggrfn not in range(5):
            raise RelalgDialectException(f"{aggrfn}: invalid aggregation function")

        col: TableColumn = column

        #if aggrfn == 3:
        #    col = TableColumn(column.scope, f"__float_{column.name}", return_type)
        #    operand = self.map(
        #        table_context=table_context,
        #        operand=operand,
        #        compute_cols=cast_to_float_region,
        #        new_columns=[col],
        #        add_new_cols_to_context=False
        #    )

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("fn", RelalgAggrFuncAttr(aggrfn)),
                ma.AttributeEntry("attr", ColumnRefAttr(col.to_symbol_ref_attr()))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def basetable(self, table_context: TableContext, table_metadata: Dict[str, Any]) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.basetable")
        return_type: ma.Type = RelalgTupleStreamType()

        op: GenericOperation = GenericOperation(
            name=name,
            args=[],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("columns", ma.DictionaryAttr([
                    ma.AttributeEntry(QuotedStr(col.name), ColumnDefAttr((
                        col.to_symbol_ref_attr(), col.type
                    )))
                    for col in table_context.cols
                ])),
                ma.AttributeEntry("table_identifier", StringAttr(table_context.id)),
                ma.AttributeEntry("meta", TableMetaDataAttr(
                    QuotedStr(json.dumps(table_metadata).replace('"', r'\22'))
                ))
            ]),
            type=[ma.FunctionType([], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def count(self, operand: ReturnValue) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.count")
        return_type: ma.Type = ma.SignlessIntegerType(64)

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def getcol(self, operand: ReturnValue, column: TableColumn) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.getcol")
        return_type: ma.Type = column.type

        if not isinstance(operand.type, RelalgTupleType):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("attr", ColumnRefAttr(column.to_symbol_ref_attr()))
            ]),
            type=[ma.FunctionType([operand.type], [column.type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    #def getlist(self, operand: ReturnValue, columns: List[TableColumn]) -> ReturnValue:
    #    name: QuotedStr = QuotedStr("relalg.getlist")
    #    return_types: List[ma.Type] = [col.type for col in columns]

    #    if not isinstance(operand.type, RelalgTupleType):
    #        raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

    #    op: GenericOperation = GenericOperation(
    #        name=name,
    #        args=[operand.sym],
    #        successors=None,
    #        regions=None,
    #        attributes=ma.AttributeDict([
    #            ma.AttributeEntry("cols", ma.ArrayAttr([
    #                ColumnRefAttr(col.to_symbol_ref_attr()) for col in columns
    #            ]))
    #        ]),
    #        type=[ma.FunctionType([operand.type], return_types)]
    #    )

    #    return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def limit(self, operand: ReturnValue, count: int) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.limit")
        return_type: ma.Type = RelalgTupleStreamType()

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("rows", ma.IntegerAttr(count, ma.SignlessIntegerType(32)))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def map(
        self,
        table_context: TableContext,
        operand: ReturnValue,
        compute_cols: ma.Region,
        new_columns: List[TableColumn],
        add_new_cols_to_context: bool = True
    ) -> ReturnValue:
        """relalg.map.

        Note that the order of the results of the region and the columns
        in the new_columns list need to match!
        """
        name: QuotedStr = QuotedStr("relalg.map")
        return_type: ma.Type = RelalgTupleStreamType()

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=[compute_cols],
            attributes=ma.AttributeDict([
                ma.AttributeEntry("computed_cols", ma.ArrayAttr([
                    ColumnDefAttr((col.to_symbol_ref_attr(), col.type)) for col in new_columns
                ]))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        if add_new_cols_to_context:
            table_context.cols.update(new_columns)

        return ReturnValue(
            self.core_builder._insert_op_in_block([None], op), return_type
        )

    def materialize(self, operand: ReturnValue, columns: List[TableColumn]) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.materialize")
        return_type: ma.Type = DBTableType()

        # TODO: columns with the same name but different scopes.
        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("cols", ma.ArrayAttr([
                    ColumnRefAttr(col.to_symbol_ref_attr()) for col in columns
                ])),
                ma.AttributeEntry("columns", ma.ArrayAttr([
                    StringAttr(col.name) for col in columns
                ]))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def merge(
        self,
        table_context: TableContext,
        merge_op: str,
        left: ReturnValue,
        right: ReturnValue,
        predicate: Optional[ma.Region] = None
    ) -> ReturnValue:
        name: QuotedStr = QuotedStr(merge_op)
        return_type: ma.Type = RelalgTupleStreamType()
        attributes: Optional[ma.AttributeDict] = None

        if merge_op == "relalg.outerjoin":
            # Currently, all columns are nullable anyway.
            attributes = ma.AttributeDict([
                ma.AttributeEntry("mapping", ma.ArrayAttr([
                    ColumnRenameDefAttr(
                        new=tc.to_symbol_ref_attr(),
                        type=tc.type,
                        old=ColumnRefAttr(tc.to_symbol_ref_attr())
                    )
                    for tc in table_context.cols
                ]))
            ])

        op: GenericOperation = GenericOperation(
            name=name,
            args=[left.sym, right.sym],
            successors=None,
            regions=[predicate] if predicate is not None else None,
            attributes=attributes,
            type=[ma.FunctionType([left.type, right.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def projection(
        self,
        table_context: TableContext,
        operand: ReturnValue,
        columns: Iterable[TableColumn],
        distinct: bool = False,
        index_cols: Optional[List[TableColumn]] = None
    ) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.projection")
        return_type: ma.Type = RelalgTupleStreamType()

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        cols: List[TableColumn] = list(columns)

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("cols", ma.ArrayAttr([
                    ColumnRefAttr(col.to_symbol_ref_attr())
                    for col in (cols + index_cols if index_cols else cols)
                ])),
                ma.AttributeEntry("set_semantic", RelalgSetSemanticAttr(0 if distinct else 1))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        # Update the columns of the table context.
        table_context.cols = OrderedSet(cols)

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def renaming(
        self,
        table_context: TableContext,
        operand: ReturnValue,
        mapper: OrderedDict[TableColumn, TableColumn]
    ) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.renaming")
        return_type: ma.Type = RelalgTupleStreamType()

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("columns", ma.ArrayAttr([
                    ColumnRenameDefAttr(
                        new=tc_new.to_symbol_ref_attr(),
                        type=tc_old.type,
                        old=ColumnRefAttr(tc_old.to_symbol_ref_attr())
                    )
                    for tc_old, tc_new in mapper.items()
                ]))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def return_op(self, results: List[ReturnValue]) -> None:
        name: QuotedStr = QuotedStr("relalg.return")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[res.sym for res in results],
            successors=None,
            regions=None,
            attributes=None,
            type=[ma.FunctionType([res.type for res in results], [])]
        )
        self.core_builder._insert_op_in_block([], op)

        return None

    def selection(self, operand: ReturnValue, predicate_ops: ma.Region) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.selection")
        return_type: ma.Type = RelalgTupleStreamType()

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=[predicate_ops],
            attributes=None,
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)

    def sort(
        self,
        operand: ReturnValue,
        sort_specs: List[Tuple[TableColumn, Literal[0, 1]]]
    ) -> ReturnValue:
        name: QuotedStr = QuotedStr("relalg.sort")
        return_type: ma.Type = RelalgTupleStreamType()

        if not self._check_tuple_stream_like(operand.type):
            raise RelalgDialectException(f"{operand.type}: invalid operand type for {name}")

        op: GenericOperation = GenericOperation(
            name=name,
            args=[operand.sym],
            successors=None,
            regions=None,
            attributes=ma.AttributeDict([
                ma.AttributeEntry("sortspecs", ma.ArrayAttr([
                    RelalgSortSpecAttr(col.to_symbol_ref_attr(), sort_spec)
                    for col, sort_spec in sort_specs
                ]))
            ]),
            type=[ma.FunctionType([operand.type], [return_type])]
        )

        return ReturnValue(self.core_builder._insert_op_in_block([None], op), return_type)
