import copy
import re
import time
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    cast
)

import mlir.astnodes as ma
import mlir.builder as mb
from pandas import Index
from pandas._typing import DtypeObj

from ._dialects.astnodes import (
    DBNullable,
    DBTableType,
    FunctionTypeAttr,
    GenericOperation,
    QuotedStr,
    RelalgTupleType,
    RelalgTupleStreamType,
    StringAttr
)
from ._dialects.base import MLIRDialectBuilder as BaseDialectBuilder
from ._dialects.common import ReturnValue, TableColumn, TableContext
from ._dialects.db import MLIRDialectBuilder as DBDialectBuilder
from ._dialects.relalg import MLIRDialectBuilder as RelalgDialectBuilder
from ._supported_types import (
    TYPES_MAPPING_DType,
    get_mlir_type_from_python_obj,
    get_mlir_value_from_python_obj
)
from ._util import OrderedDict, OrderedSet


class MLIRBuilderException(Exception):
    pass


class InvalidArgumentException(MLIRBuilderException):
    pass


class UnsupportedColumnName(MLIRBuilderException):
    pass


class UnsupportedDataType(MLIRBuilderException):
    pass


class DBArithOp(Enum):
    add    = auto()
    sub    = auto()
    mul    = auto()
    div    = auto()
    mod    = auto()
    and_op = auto()
    or_op  = auto()
    not_op = auto()


# cf. lingodb: build/lingodb-debug/include/mlir/Dialect/DB/IR/DBOpsEnums.h.inc
class DBCmpPredicate(Enum):
    EQ  = 0
    NEQ = 1
    LT  = 2
    LTE = 3
    GT  = 4
    GTE = 5


# cf. https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
# (or https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html)
#
# duplicate enum members:
# https://docs.python.org/3/library/enum.html#duplicating-enum-members-and-values
class MergeFlavor(Enum):
    left       = "relalg.outerjoin"
    right      = "relalg.outerjoin"
    outer      = "relalg.fullouterjoin"
    inner      = "relalg.join"
    cross      = "relalg.crossproduct"


# cf. lingodb: build/lingodb-debug/include/mlir/Dialect/RelAlg/IR/RelAlgOpsEnums.h.inc
class RelalgAggrFunc(Enum):
    sum   = 0
    min   = 1
    max   = 2
    mean  = 3
    count = 4
    #any   = 5 TODO?


# cf. lingodb: build/lingodb-debug/include/mlir/Dialect/RelAlg/IR/RelAlgOpsEnums.h.inc
class RelalgSortSpec(Enum):
    desc = 0
    asc  = 1


# cf. lingodb: build/lingodb-debug/include/mlir/Dialect/DB/IR/DBOpsEnums.h.inc
class TimeUnitAttr(Enum):
    SECOND      = 0
    MILLISECOND = 1
    MICROSECOND = 2
    NANOSECOND  = 3


def detach_func(attr: Callable[..., Any]) -> Any:
    """Break internal references.

    Assuming two MLIRBuilder instances sharing building on the same
    function, but having different internal states.
    This happens when the owning MLIRPandas objects are not-yet
    materialized, but independent.
    Sometimes, after operations affecting the row configuration (such
    as selection), we need to seperate the function of the current
    MLIRBuilder, so replace the reference by an own copy.
    """
    def wrapper(self: "MLIRBuilder", *args: Any, **kwargs: Any) -> Any:
        self.state = copy.deepcopy(self.state) # TODO: is this good here? maybe instead of complicated copy semantics...
        self._detach_func()
        return attr(self, *args, **kwargs)

    return wrapper


class MLIRBuilder:
    CUSTOM_DIALECTS: Final[Tuple[Tuple[str, Type[mb.builder.DialectBuilder]], ...]] = (
        ("base", BaseDialectBuilder), ("db", DBDialectBuilder), ("relalg", RelalgDialectBuilder)
    )

    class State:
        def __init__(self, return_value: ReturnValue, *args: ReturnValue) -> None:
            self.return_values: List[ReturnValue] = [return_value, *args]

        @property
        def return_value(self) -> ReturnValue:
            return self.return_values[-1]

        @return_value.setter
        def return_value(self, value: ReturnValue) -> None:
            self.return_values.append(value)

    def __init__(
        self,
        table_identifier: str,
        icols: List[TableColumn],
        columns: Optional[List[Tuple[Any, DtypeObj]]] = None,
        builder: Optional[mb.IRBuilder] = None,
        func_data: Optional[Tuple[GenericOperation, "MLIRBuilder.State"]] = None,
        table_context_cols: Optional[OrderedSet[TableColumn]] = None,
        func_defs: List[Tuple[GenericOperation, ma.Type]] = []
    ) -> None:
        """Build an MLIR module for DataFrame operations.

        Parameters:
            table_identifier    The identifier of the table.
            icols               The index columns.
            columns             A list of tuples containing the
                                columns and their corresponding type.
                                E.g.:
                                  [('col1', dtype('int64')),
                                   ('col2', dtype('O'))]
            builder             A builder object to use.
            func_data           The function to add the operations and
                                an MLIRBuilder.State object.
            table_context_cols  If not None, this is a reference to
                                an existing ordered set of TableColumns.
            func_defs           A list of function definitions (along
                                with the corresponding return type) to
                                insert into the final module.
        """
        self.func: GenericOperation
        self.state: MLIRBuilder.State

        if columns is None and table_context_cols is None:
            raise MLIRBuilderException("both columns and table_context_cols are None")
        if columns is not None and table_context_cols is not None:
            raise MLIRBuilderException("both columns and table_context_cols are given")

        if table_context_cols is None:
            try:
                table_context_cols = OrderedSet(
                    TableColumn(
                        scope=table_identifier,
                        name=str(col_name),
                        type=TYPES_MAPPING_DType[col_type].mlir
                    )
                    for col_name, col_type in columns # type: ignore
                )
            except KeyError as e:
                raise UnsupportedDataType(e.args[0])

        self.icols: List[TableColumn] = icols

        # Initialize the table context.
        self.table_context = TableContext(id=table_identifier, cols=table_context_cols)

        # Table metadata is currently unused.
        self.table_metadata: Dict[str, Any] = {}

        # Prevent the following error from the MLIR parser:
        # error: @ identifier expected to start with letter or '_'
        identifier_reg: re.Pattern[str] = re.compile("[_a-zA-Z].*")
        try:
            if not all(identifier_reg.fullmatch(col.name)
                       for col in self.table_context.cols):
                raise UnsupportedColumnName()
        except TypeError:
            raise UnsupportedColumnName()

        self.builder: mb.IRBuilder = builder if builder is not None else self.create_ir_builder()

        # Create the function for this MLIRBuilder.
        if func_data is None:
            func, state = self.create_func(
                self.builder, self.table_context, self.table_metadata, self.icols
            )
        else:
            func, state = func_data
        self.set_func(func, state)

        self._func_defs: List[Tuple[GenericOperation, ma.Type]] = func_defs

    @classmethod
    def _basetable(
        cls,
        builder: mb.IRBuilder,
        func: GenericOperation,
        table_context: TableContext,
        table_metadata: Dict[str, Any],
        icols: List[TableColumn]
    ) -> ReturnValue:
        if not func.regions:
            raise MLIRBuilderException("given function has no region")
        with builder.goto_block(func.regions[0].body[0]):
            # Temporarily include the internal index column.
            table_context.cols.update(icols)
            # Insert the operation to load the data.
            ret: ReturnValue = builder.relalg.basetable(
                table_context=table_context,
                table_metadata=table_metadata,
            )
            # Remove the index column from the context again.
            table_context.cols.difference_update(icols)
        return ret

    @classmethod
    def combine(
        cls,
        left: "MLIRBuilder",
        right: "MLIRBuilder",
        table_identifier: str,
        table_context_cols: OrderedSet[TableColumn],
        use_left_icols: bool = True
    ) -> "MLIRBuilder":
        builder: mb.IRBuilder = cls.create_ir_builder()

        main: GenericOperation = builder.base.func(name="main", return_types=[DBTableType()])
        func_left_name: str = f"func_{time.time_ns()}"
        func_right_name: str = f"func_{time.time_ns()}"

        func_left: GenericOperation = copy.deepcopy(left.func)
        func_left.attributes = ma.AttributeDict([
            ma.AttributeEntry("sym_name", StringAttr(func_left_name)),
            ma.AttributeEntry("function_type", FunctionTypeAttr(
                ma.FunctionType([], [left.state.return_value.type])
            )),
            ma.AttributeEntry("sym_visibility", StringAttr("private"))
        ])
        func_right: GenericOperation = copy.deepcopy(right.func)
        func_right.attributes = ma.AttributeDict([
            ma.AttributeEntry("sym_name", StringAttr(func_right_name)),
            ma.AttributeEntry("function_type", FunctionTypeAttr(
                ma.FunctionType([], [right.state.return_value.type])
            )),
            ma.AttributeEntry("sym_visibility", StringAttr("private"))
        ])

        if not main.regions or not func_left.regions or not func_right.regions:
            raise MLIRBuilderException("created function without region")

        with builder.goto_block(func_left.regions[0].body[0]):
            builder.base.return_op([left.state.return_value])
        with builder.goto_block(func_right.regions[0].body[0]):
            builder.base.return_op([right.state.return_value])

        with builder.goto_block(main.regions[0].body[0]):
            return_value_left: ReturnValue = builder.base.call(
                func_name=func_left_name, return_types=[left.state.return_value.type]
            )
            return_value_right: ReturnValue = builder.base.call(
                func_name=func_right_name, return_types=[right.state.return_value.type]
            )

        ret: "MLIRBuilder" = MLIRBuilder(
            table_identifier=table_identifier,
            icols=(left if use_left_icols else right).icols,
            builder=builder,
            func_data=(main, cls.State(return_value_left, return_value_right)),
            table_context_cols=table_context_cols,
            func_defs=left._func_defs + right._func_defs + [
                (func_left, left.state.return_value.type),
                (func_right, right.state.return_value.type)
            ]
        )
        return ret

    @classmethod
    def create_func(
        cls,
        builder: mb.IRBuilder,
        table_context: TableContext,
        table_metadata: Dict[str, Any],
        icols: List[TableColumn]
    ) -> Tuple[GenericOperation, State]:
        """Create the main function for an MLIRBuilder instance."""
        func: GenericOperation = builder.base.func(name="main", return_types=[DBTableType()])
        ret: ReturnValue = cls._basetable(builder, func, table_context, table_metadata, icols)
        return (func, cls.State(ret))

    @classmethod
    def create_ir_builder(cls) -> mb.IRBuilder:
        builder: mb.IRBuilder = mb.IRBuilder()
        # Register dialects.
        for dialect_name, dialect_builder in cls.CUSTOM_DIALECTS:
            builder.register_dialect(dialect_name, dialect_builder(builder))
        return builder

    @classmethod
    def from_other(
        cls,
        other: "MLIRBuilder",
        table_identifier: str,
        cols: OrderedSet[TableColumn],
        do_copy: bool = True
    ) -> "MLIRBuilder":
        """Create an MLIRBuilder instance from another one.

        If do_copy is not True, both MLIRBuilder objects will work with
        the same builder on the same function.
        """
        ret: "MLIRBuilder" = cls(
            table_identifier=table_identifier,
            icols=other.icols,
            builder=None if do_copy else other.builder,
            func_data=(
                (other.func, copy.deepcopy(other.state))
                if do_copy else (other.func, other.state) # copy.deepcopy(other.state))
            ),
            table_context_cols=cols,
            func_defs=other._func_defs
        )
        if do_copy: # and cols:
            ret.copy_scope(ret.table_context.id)
        return ret

    @property
    def col_config(self) -> OrderedSet[TableColumn]:
        """Return the current column configuration.

        Only return the columns of the (mayble larger) underlying data,
        which are handled by the object this builder belongs to.
        """
        return self.table_context.cols

    @property
    def need_materialize(self) -> bool:
        # Check whether there are more operations than the first "relalg.basetable".
        if (not self.func.regions
                or not self.func.regions[0].body
                or not self.func.regions[0].body[0]
                or not self.func.regions[0].body[0].body):
            return False
        return len(self.func.regions[0].body[0].body) > 1

    def _detach_func(self) -> None:
        self.set_func(copy.deepcopy(self.func), self.state)

    #@detach_func
    def arith_cols(
        self,
        op: DBArithOp,
        col: Optional[TableColumn],
        columns: OrderedSet[TableColumn],
        result_name: Optional[str] = None
    ) -> None:
        """Do an arithmetic operation on a constant and the given columns.

        Check that the types match. If this is not the case, throw an
        InvalidArgumentException.
        """
        # Check the types. TODO: support more scenarios?
        if col is not None:
            for c in columns:
                if c.type != col.type:
                    raise InvalidArgumentException(f"{c.type} does not match {col.type}")

        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            ret_col: Optional[ReturnValue] = (
                self.builder.relalg.getcol(row, col) if col is not None else None
            )
            # TODO: do getlist and the loop in MLIR?
            ret_cols: List[ReturnValue] = [
                self.builder.relalg.getcol(row, col) for col in columns
            ]
            results: List[ReturnValue] = [
                getattr(self.builder.db, op.name)(ret_c, ret_col)
                for ret_c in ret_cols
            ]
            # Return the new values.
            self.builder.relalg.return_op(results)

        new_columns: List[TableColumn]
        if col is not None:
            if result_name is None:
                raise MLIRBuilderException("cannot add column without a name")
            new_columns = [TableColumn(self.table_context.id, result_name, col.type)]
        else:
            new_columns = [TableColumn(self.table_context.id, tc.name, tc.type) for tc in columns]

        self.state.return_value = self.builder.relalg.map(
            table_context=self.table_context,
            operand=self.state.return_value,
            compute_cols=region,
            new_columns=new_columns
        )

    #@detach_func
    def arith_constant(
        self,
        op: DBArithOp,
        constant: Any,
        columns: OrderedSet[TableColumn],
        result_name: Optional[str] = None
    ) -> None:
        """Do an arithmetic operation on a constant and the given columns.

        Check that the types match. If this is not the case, throw an
        InvalidArgumentException.
        """
        mlir_type: ma.Type = get_mlir_type_from_python_obj(constant)

        # Check the types. TODO: support more scenarios?
        for col in columns:
            if col.type != mlir_type:
                raise InvalidArgumentException(f"{col.type} does not match {mlir_type}")

        if isinstance(mlir_type, DBNullable):
            mlir_type = mlir_type.real_type

        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            ret_constant: ReturnValue = self.builder.db.constant(
                ma.PrimitiveAttribute(constant, mlir_type)
            )
            # TODO: currently assume everything is nullable.
            ret_constant_nullable: ReturnValue = self.builder.db.as_nullable(ret_constant)
            # TODO: do getlist and the loop in MLIR?
            ret_cols: List[ReturnValue] = [
                self.builder.relalg.getcol(row, col) for col in columns
            ]
            results: List[ReturnValue] = [
                getattr(self.builder.db, op.name)(ret_col, ret_constant_nullable)
                for ret_col in ret_cols
            ]
            # Return the new values.
            self.builder.relalg.return_op(results)

        new_columns: List[TableColumn]
        if result_name is not None:
            if len(columns) > 1:
                raise MLIRBuilderException("result_name currently only suitable for one column")
            new_columns = [
                TableColumn(self.table_context.id, result_name, tc.type) for tc in columns
            ]
        else:
            new_columns = [TableColumn(self.table_context.id, tc.name, tc.type) for tc in columns]

        self.state.return_value = self.builder.relalg.map(
            table_context=self.table_context,
            operand=self.state.return_value,
            compute_cols=region,
            new_columns=new_columns
        )

    @detach_func
    def aggregate(
        self,
        group_by_cols: List[TableColumn],
        cols_mapping: Dict[str, Tuple[TableColumn, RelalgAggrFunc]]
    ) -> None:
        stream: ReturnValue = ReturnValue(
            ma.SsaId(f"_arg_stream_{time.time_ns()}"), RelalgTupleStreamType()
        )
        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([stream, row])

        with self.builder.goto_block(region.body[0]):
            ret_aggr: List[ReturnValue] = []
            for col, aggrfn in cols_mapping.values():
                if col in group_by_cols:
                    if aggrfn != RelalgAggrFunc.count:
                        raise MLIRBuilderException("tried to aggregate group key")
                    ret_aggr.append(self.builder.relalg.count(stream))
                else:
                    ret_aggr.append(
                        self.builder.relalg.aggrfn(
                            operand=stream,
                            aggrfn=aggrfn.value,
                            column=col,
                            #table_context=self.table_context,
                            #cast_to_float_region=self.get_cast_to_float_region(col)
                        )
                    )
            self.builder.relalg.return_op(ret_aggr)

        self.state.return_value = self.builder.relalg.aggregation(
            table_context=self.table_context,
            operand=self.state.return_value,
            group_by_cols=group_by_cols,
            computed_cols=[
                TableColumn(self.table_context.id, col_name, ret.type)
                for (col_name, _), ret in zip(cols_mapping.items(), ret_aggr)
            ],
            region=region
        )

        # Remove the columns which have been aggregated in the new groups.
        #self.table_context.cols.difference_update(col for col, _ in cols_mapping)
        # TODO: what happens here?

    #@detach_func
    def compare_cols(
        self,
        left: TableColumn,
        right: TableColumn,
        op: DBCmpPredicate,
        new_column: str,
        new_scope: str
    ) -> None:
        # Check the types. TODO: support more scenarios?
        if left.type != right.type:
            raise InvalidArgumentException(f"{left.type} does not match {right.type}")

        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            ret_col_left: ReturnValue = self.builder.relalg.getcol(row, left)
            ret_col_right: ReturnValue = self.builder.relalg.getcol(row, right)
            ret_comp: ReturnValue = self.builder.db.compare(ret_col_left, ret_col_right, op.value)
            self.builder.relalg.return_op([ret_comp])

        new_tc: TableColumn = TableColumn(new_scope, new_column, ret_comp.type)

        self.state.return_value = self.builder.relalg.map(
            table_context=self.table_context,
            operand=self.state.return_value,
            compute_cols=region,
            new_columns=[new_tc]
        )

        self.table_context.cols.add(new_tc)

    #@detach_func
    def compare_constant(
        self,
        column: TableColumn,
        constant: Any,
        op: DBCmpPredicate,
        new_column: str,
        new_scope: str
    ) -> None:
        mlir_type: ma.Type = get_mlir_type_from_python_obj(constant)

        # Check the types. TODO: support more scenarios?
        if column.type != mlir_type:
            raise InvalidArgumentException(f"{column.type} does not match {mlir_type}")

        if isinstance(mlir_type, DBNullable):
            mlir_type = mlir_type.real_type

        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            ret_constant: ReturnValue = self.builder.db.constant(
                ma.PrimitiveAttribute(get_mlir_value_from_python_obj(constant), mlir_type)
            )
            # TODO: currently assume everything is nullable.
            ret_constant_nullable: ReturnValue = self.builder.db.as_nullable(ret_constant)
            ret_col: ReturnValue = self.builder.relalg.getcol(row, column)
            ret_comp: ReturnValue = self.builder.db.compare(
                ret_col, ret_constant_nullable, op.value
            )
            self.builder.relalg.return_op([ret_comp])

        new_tc: TableColumn = TableColumn(new_scope, new_column, ret_comp.type)

        self.state.return_value = self.builder.relalg.map(
            table_context=self.table_context,
            operand=self.state.return_value,
            compute_cols=region,
            new_columns=[new_tc]
        )

        self.table_context.cols.add(new_tc)

    def copy_scope(self, new_scope: str, old_scope: Optional[str] = None) -> None:
        """Copy all columns of a scope to another scope.

        Also replace them in the columns configuration (accessible via
        the _col_config property).
        """
        old_columns: List[TableColumn] = [
            tc for tc in self.table_context.cols
            if old_scope is None or tc.scope == old_scope
        ]
        old_icols: List[TableColumn] = self.icols

        # The order of the two columns lists need to match!
        new_columns: List[TableColumn] = [
            TableColumn(new_scope, tc.name, tc.type) for tc in old_columns
        ]
        new_icols: List[TableColumn] = [
            TableColumn(new_scope, tc.name, tc.type) for tc in old_icols
        ]

        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            # TODO: do getlist and the loop in MLIR?
            ret_cols: List[ReturnValue] = [
                self.builder.relalg.getcol(row, col) for col in old_columns + old_icols
            ]
            self.builder.relalg.return_op(ret_cols)

        self.state.return_value = self.builder.relalg.map(
            table_context=self.table_context,
            operand=self.state.return_value,
            compute_cols=region,
            new_columns=new_columns + new_icols
        )
        # map adds all added columns to the table columns!
        # Remove the index cols again.
        self.table_context.cols.difference_update(new_icols)
        # Also remove the old columns.
        self.table_context.cols.difference_update(old_columns)
        self.icols = new_icols

        self.projection(self.table_context.cols)

    @detach_func
    def drop_duplicates(self) -> None:
        self.projection(self.table_context.cols, distinct=True)

    @detach_func
    def dropna(self, cols: List[TableColumn]) -> None:
        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            ret_cols: List[ReturnValue] = [self.builder.relalg.getcol(row, col) for col in cols]
            ret_is_non_null: List[ReturnValue] = [
                self.builder.db.not_op(self.builder.db.isnull(ret_col))
                for ret_col in ret_cols
            ]
            self.builder.relalg.return_op([ret_is_non_null])

        self.state.return_value = self.builder.relalg.selection(
            operand=self.state.return_value,
            predicate_ops=region
        )

    def get_cast_to_float_region(self, col: TableColumn) -> ma.Region:
        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            ret_col: ReturnValue = self.builder.relalg.getcol(row, col)
            ret_cast: ReturnValue = self.builder.db.cast(
                ret_col, DBNullable(ma.FloatType(ma.FloatTypeEnum.f64))
            )
            self.builder.relalg.return_op([ret_cast])

        return region

    def groupby(self, cols: List[TableColumn]) -> None:
        self.state.return_value = self.builder.relalg.aggregation(
            operand=self.state.return_value,
            group_by_cols=cols,
        )

    @detach_func
    def limit(self, count: int) -> None:
        self.state.return_value = self.builder.relalg.limit(
            operand=self.state.return_value, count=count
        )

    @detach_func
    def materialize(self, cols: OrderedSet[TableColumn]) -> str:
        """Build the MLIR module and return it as string.

        Only materialize the given columns, specified as tuples of
        (scope, column name) values.

        This method may be called multiple times on the same object!
        """
        # Get the return value from the last operation (consider that
        # multiple MLIRBuilder objects may build on the same MLIR module!)
        # TODO: Currently we assume that the last ssa id is not a list
        # and that it is of the right type.
        if self.builder.block is None:
            raise MLIRBuilderException("builder block is None - this should not happen")
        # Note: Despite being type hinted otherwise, the "value" of an "OpResult"
        # is a string, not an SsaId.
        ret: ReturnValue = ReturnValue(
            ma.SsaId(cast(str, self.builder.block.body[-1].result_list[0].value)),
            RelalgTupleStreamType()
        )
        # Insert a relalg materialize operation.
        ret = self.builder.relalg.materialize(operand=ret, columns=cols)
        # Finish with the return operation.
        self.builder.base.return_op([ReturnValue(ret.sym, ret.type)])

        # Create an MLIR file and insert the main-function.
        mlirfile: ma.MLIRFile = self.builder.make_mlir_file(self.builder.base.module())
        with self.builder.goto_block(mlirfile.default_module.region.body[0]):
            seen: Set[str] = set()
            for func, _ in self._func_defs:
                if func.attributes is None:
                    raise MLIRBuilderException("func has no attributes")
                sym_attrs: List[ma.AttributeEntry] = [
                    attr for attr in func.attributes.values if attr.name == "sym_name"
                ]
                if len(sym_attrs) != 1:
                    raise MLIRBuilderException("func has no or ambiguous name")
                name: str = cast(QuotedStr, sym_attrs[0].value).value
                if name in seen:
                    continue
                self.builder._insert_op_in_block([], func)
                seen.add(name)
            self.builder._insert_op_in_block([], self.func)

        try:
            # Dump the file as string.
            mlir: str = mlirfile.dump()
        except Exception as e:
            raise MLIRBuilderException(
                "an exception occurred while dumping the MLIRFile: %s" % str(e)
            )

        # Remove the added operations from the function.
        if not self.func.regions:
            raise MLIRBuilderException("given function has no region")
        self.func.regions[0].body[0].body.clear()
        # Insert the basetable operation again.
        self.state.return_value = self._basetable(
            self.builder, self.func, self.table_context, self.table_metadata, self.icols
        )
        # Reposition the builder.
        self.builder.position_at_exit(self.func.regions[0].body[0])
        # Remove the other function definitions.
        self._func_defs.clear()

        return mlir

    def merge(
        self,
        merge_flavor: MergeFlavor,
        left_on: List[TableColumn],
        right_on: List[TableColumn]
    ) -> None:
        if len(self.state.return_values) < 2:
            raise MLIRBuilderException("not enough return values for merge")
        if len(left_on) != len(right_on):
            raise MLIRBuilderException("number of left and right columns to merge on is not equal")
        # TODO: support more scenarios?
        if any(c.type != d.type for c, d in zip(left_on, right_on)):
            raise MLIRBuilderException(f"cannot compare columns of types {left_on}, {right_on}")

        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            left_cols: List[ReturnValue] = [
                self.builder.relalg.getcol(row, col) for col in left_on
            ]
            right_cols: List[ReturnValue] = [
                self.builder.relalg.getcol(row, col) for col in right_on
            ]
            ret_comp: List[ReturnValue] = [
                self.builder.db.compare(left_col, right_col, DBCmpPredicate.EQ.value)
                for left_col, right_col in zip(left_cols, right_cols)
            ]
            self.builder.relalg.return_op(ret_comp)

        self.state.return_value = self.builder.relalg.merge(
            table_context=self.table_context,
            merge_op=merge_flavor.value,
            left=self.state.return_values[-2],
            right=self.state.return_values[-1],
            predicate=region
        )

        if merge_flavor == MergeFlavor.outer:
            self.state.return_value = self.builder.relalg.sort(
                operand=self.state.return_value,
                sort_specs=[(col, RelalgSortSpec.asc) for col in left_on + right_on]
            )

    def projection(
        self,
        columns: OrderedSet[TableColumn],
        distinct: bool = False,
        keep_icols: bool = True
    ) -> None:
        # Note that we need to keep the order of the column names!
        # TODO: no. why?
        self.state.return_value = self.builder.relalg.projection(
            table_context=self.table_context,
            operand=self.state.return_value,
            columns=columns,
            distinct=distinct,
            index_cols=self.icols if keep_icols else None
        )

    def rename(self, mapper: OrderedDict[TableColumn, TableColumn]) -> None:
        self.state.return_value = self.builder.relalg.renaming(
            table_context=self.table_context,
            operand=self.state.return_value,
            mapper=mapper
        )
        # Keep the order of the columns.
        self.table_context.cols = OrderedSet(
            tc if not tc in mapper else mapper[tc]
            for tc in self.table_context.cols
        )

    @detach_func
    def selection_col(self, constant: Any, column: TableColumn) -> None:
        """Selection according to value in given column."""
        mlir_type: ma.Type = get_mlir_type_from_python_obj(constant)

        # Check the types. TODO: support more scenarios?
        if column.type != mlir_type:
            raise InvalidArgumentException(f"{column.type} does not match {mlir_type}")

        if isinstance(mlir_type, DBNullable):
            mlir_type = mlir_type.real_type

        row: ReturnValue = ReturnValue(ma.SsaId(f"_arg_row_{time.time_ns()}"), RelalgTupleType())
        region: ma.Region = create_region([row])

        with self.builder.goto_block(region.body[0]):
            ret_col: ReturnValue = self.builder.relalg.getcol(row, column)
            if (ret_col.type != ma.SignlessIntegerType(1)
                    and ret_col.type != DBNullable(ma.SignlessIntegerType(1))):
                ret_is_non_null: ReturnValue = self.builder.db.not_op(
                    self.builder.db.isnull(ret_col)
                )
                ret_constant: ReturnValue = self.builder.db.constant(
                    ma.PrimitiveAttribute(get_mlir_value_from_python_obj(constant), mlir_type)
                )
                ret_comp: ReturnValue = self.builder.db.compare(
                    ret_col, ret_constant, DBCmpPredicate.EQ.value
                )
                # Check that the column value is not null.
                ret_col = self.builder.db.and_op(ret_comp, ret_is_non_null)
            self.builder.relalg.return_op([ret_col])

        self.state.return_value = self.builder.relalg.selection(
            operand=self.state.return_value,
            predicate_ops=region
        )

    def set_func(self, func: GenericOperation, state: "MLIRBuilder.State") -> None:
        self.func = func
        # Start building in our new function.
        if not self.func.regions:
            raise MLIRBuilderException("given function has no region")
        self.builder.position_at_exit(self.func.regions[0].body[-1])
        self.state = state

    @detach_func
    def sort(self, sort_specs: List[Tuple[TableColumn, RelalgSortSpec]]) -> None:
        self.state.return_value = self.builder.relalg.sort(
            operand=self.state.return_value,
            sort_specs=[(col, sort_spec.value) for col, sort_spec in sort_specs]
        )


def create_icol(scope: str, index_type: DtypeObj, index_name: str = "") -> List[TableColumn]:
    # TODO: can support multi-index
    return [TableColumn(
        scope,
        ICOL_NAME + index_name,
        TYPES_MAPPING_DType[index_type].mlir
    )]


def create_region(block_args: List[ReturnValue], block_id: str = "bb0") -> ma.Region:
    return ma.Region([
        ma.Block(
            label=ma.BlockLabel(
                name=ma.BlockId(block_id),
                arg_ids=[arg.sym for arg in block_args],
                arg_types=[arg.type for arg in block_args]
            ),
            body=[]
        )
    ])


# Information about the internal index column.
ICOL_NAME: Final[str] = "__index__"
#ICOL: Final[TableColumnSL] = TableColumnSL("__index__", ma.SignlessIntegerType(64))
