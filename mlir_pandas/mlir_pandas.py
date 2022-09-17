from dataclasses import dataclass
import itertools
import logging
import os
import sys
import time
from types import SimpleNamespace as Namespace
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pyarrow as pa
import pandas as pd
import pandas.core.groupby as pd_gb
from pandas._typing import DtypeObj
# TODO: why do we need this?
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
from pymlirdbext import load as db_load, run as db_run

from ._dialects.common import TableColumn
from ._mlir_builder import (
    DBArithOp,
    DBCmpPredicate,
    InvalidArgumentException,
    MergeFlavor,
    MLIRBuilder,
    MLIRBuilderException,
    RelalgAggrFunc,
    RelalgSortSpec,
    UnsupportedColumnName,
    UnsupportedDataType,
    create_icol
)
from ._supported_types import (
    PyArrow2PandasConversionArgs,
    PyArrowFromPandasConversionArgs,
    SupportedTypesException,
    get_dtype_from_pyarrow_type
)
from ._util import OrderedDict, OrderedSet, gettime


ArrayLikeType = TypeVar("ArrayLikeType", bound="ArrayLike")
MLIRPandasType = TypeVar("MLIRPandasType", bound="MLIRPandas")
PandasType = Union[pd.DataFrame, pd_gb.DataFrameGroupBy, pd.Series]


class MLIRPandasException(Exception):
    pass


class OverrideException(Exception):
    pass


def override(original_name: str, ret_type: Optional[str] = None, mock: bool = False) -> Any:
    """Handle overriding of attributes of the original class.

    In case the custom method cannot handle a given scenario, it is
    expected to throw an OverrideException. This will be caught here
    and the original (overriden) method will be called instead.

    Take care that the return type always gets converted to this
    class when it is of type PARENTCLASS.

    Parameters:
    -----------
    original_name    The original name of the attribute that gets
                     overwritten.
    ret_type         The expected return type.
                     This needs to be provided as string in order to
                     avoid cyclic dependencies between the classes in
                     this module.
    """
    def _override_inner(attr: Any) -> Any:
        def wrapper_callable(self: "MLIRPandas", *args: Any, **kwargs: Any) -> Any:
            result: Any
            if self._mlir_builder is not None:
                try:
                    result = attr(self, *args, **kwargs)
                except OverrideException:
                    result = getattr(self._mock, original_name)(*args, **kwargs)
            else:
                result = getattr(self._mock, original_name)(*args, **kwargs)
            return self._construct_return_value(
                result, _CLASSES[ret_type] if ret_type is not None else None, mock
            )

        def wrapper(self: "MLIRPandas") -> Any:
            my_attr: Any = attr
            if self._mlir_builder is None:
                my_attr = getattr(self._mock, original_name)
            return self._construct_return_value(
                my_attr, _CLASSES[ret_type] if ret_type is not None else None, mock
            )

        return wrapper_callable if callable(attr) else wrapper

    return _override_inner


@dataclass
class Data:
    """Holds the mocked data."""
    mock: Any

    def get_index_info(self) -> Tuple[DtypeObj, str]:
        if self.is_arrow():
            return (pd.Int64Dtype(), "")
        return (self.mock.index.dtype, str(self.mock.index.name))

    def is_arrow(self) -> bool:
        return isinstance(self.mock, pa.Table)

    def to_arrow(self, icol_name: str) -> pa.Table:
        table: pa.Table
        df: Optional[pd.DataFrame] = None

        if self.is_arrow():
            table = self.mock
        else:
            if isinstance(self.mock, pd.DataFrame):
                df = DataFrame._to_df(self.mock)
            elif isinstance(self.mock, pd_gb.DataFrameGroupBy):
                df = DataFrameGroupBy._to_df(self.mock)
            elif isinstance(self.mock, pd.Series):
                df = Series._to_df(self.mock)
            else:
                raise MLIRPandasException(
                    f"cannot convert obj of type {type(self.mock)} to pd.DataFrame"
                )
            table = pa.Table.from_pandas(df, **PyArrowFromPandasConversionArgs)

        # Add the origial pandas index.
        # TODO: can support multi-index.
        ret = table.append_column(icol_name, [np.arange(table.num_rows) if df is None else df.index])
        return ret


class MLIRPandas:
    _FINISHED_INIT: Final[str] = "_finished_init"
    # The class represented by the current class.
    _MOCKCLASS: Type[Any]
    _DEFAULT_KWARGS: Dict[str, Any] = {}

    def __init__(
        self,
        *args: Any,
        _mock_: Optional[Any] = None,
        _timestamps_: List[Tuple[str, int]] = [],
        **kwargs: Any
    ) -> None:
        if _mock_ is not None and not _timestamps_:
            logging.debug("_mock_ is given, but _timestamps_ not")
        self._timestamps: List[Tuple[str, int]] = _timestamps_ + [("init", gettime())]
        # TODO: handle specific args and kwargs?
        self._timestamps.append(("pre mock init", gettime()))
        self._mock_: Optional[Any] = Data(_mock_ if _mock_ is not None else (
            self._MOCKCLASS(*args, **kwargs)
            if args or kwargs else self._MOCKCLASS(**self._DEFAULT_KWARGS)
        ))
        self._timestamps.append(("post mock init", gettime()))
        # Store references to Data objects and the corresponding table identifiers.
        # There may be multiple reference if this object arose from combining
        # others by a join, for example.
        self._orig_references: Dict[str, Data] = {}
        # TODO: the current implementation of a unique id is just for prototyping
        # Also "_table_identifier" would maybe better be named "_scope".
        self._table_identifier: str = f"data_{time.time_ns()}"
        # _table_identifier needs to be defined before _mlir_builder.
        self._mlir_builder: Optional[MLIRBuilder]
        try:
            index_type, index_name = self._mock_.get_index_info()
            self._mlir_builder = self._mlir_builder_create(
                self._mock_, create_icol(self._table_identifier, index_type, index_name)
            )
        except:
            logging.debug("failed to create MLIRBuilder; mlir_builder will be None")
            self._mlir_builder = None
        # Some custom initialisation.
        self._init_custom()
        # After this statement, all attributes added or changed get
        # propagated to the underlying data object.
        # (In order to make exceptions for a certain method, delete and
        # add self._FINISHED_INIT again.)
        setattr(self, self._FINISHED_INIT, None)

    def _init_custom(self) -> None:
        pass

    @staticmethod
    def _same_table(left: MLIRPandasType, right: MLIRPandasType) -> bool:
        # Try to determine whether the columns of left and right are
        # currently located in the same table.
        if left._mlir_builder is None or right._mlir_builder is None:
            return False
        return left._mlir_builder.icols == right._mlir_builder.icols
        #if not left._orig_references and not right._orig_references:
        #    return False
        #return (
        #    (
        #        right._table_identifier in left._orig_references
        #        and len(left._orig_references) == 1
        #    ) or (
        #        left._table_identifier in right._orig_references
        #        and len(right._orig_references) == 1
        #    ) or left._orig_references == right._orig_references
        #)

    @staticmethod
    def _to_df(_: PandasType) -> pd.DataFrame:
        raise NotImplementedError()

    @classmethod
    def _construct_return_value(
        cls,
        ret_value: Any,
        ret_type: Optional[Type[Any]] = None,
        mock: bool = False
    ) -> Any:
        if ret_type is not None:
            return ret_value if isinstance(ret_value, ret_type) else (
                ret_type(_mock_=ret_value) if mock else ret_type(ret_value)
            )
        for orig_type, fake_type in _MOCK_MAP.items():
            if isinstance(ret_value, orig_type):
                return fake_type(_mock_=ret_value)
        return ret_value

    @classmethod
    def _init_internal(
        cls: Type[MLIRPandasType],
        mock: Optional[PandasType],
        orig_references: Dict[str, Data],
        mlir_builder: Optional[MLIRBuilder],
        table_identifier: str = f"data_{time.time_ns()}",
        other_mlir_builder: Optional[MLIRBuilder] = None,
        table_context_cols: OrderedSet[TableColumn] = OrderedSet(),
        timestamps: List[Tuple[str, int]] = []
    ) -> MLIRPandasType:
        """Does not call __init__, but provides an own construction procedure."""
        ret: MLIRPandasType = cls.__new__(cls)
        ret._timestamps = timestamps + [("init", gettime())]
        ret._mock_ = Data(mock) if mock is not None else None
        ret._orig_references = orig_references
        ret._table_identifier = table_identifier
        if mlir_builder is not None and other_mlir_builder is not None:
            ret._mlir_builder = MLIRBuilder.combine(
                left=mlir_builder,
                right=other_mlir_builder,
                table_identifier=table_identifier,
                table_context_cols=table_context_cols
            )
        else:
            ret._mlir_builder = mlir_builder
        ret._init_custom()
        setattr(ret, ret._FINISHED_INIT, None)
        return ret

    @classmethod
    def from_pyarrow_table(cls: Type[MLIRPandasType], table: pa.Table) -> MLIRPandasType:
        return cls(_mock_=table)

    @property
    def _icol(self) -> TableColumn:
        return self._mlir_builder_safe.icols[0]

    @_icol.setter
    def _icol(self, col: TableColumn) -> None:
        self._mlir_builder_safe.icols[0] = col

    @property
    def _mock(self) -> Any:
        """Materialize."""
        if (self._mlir_builder is not None
                and (self._mlir_builder.need_materialize or self._mock_ is None)):
            self._execute_mlir()
        if self._mock_ is not None and self._mock_.is_arrow():
            self._timestamps.append(("pre load from db", gettime()))
            self._load_data_from_db(self._mock_.mock, ignore_index=True)
            self._timestamps.append(("post load from db", gettime()))
        return self._mock_.mock if self._mock_ is not None else None

    # cf. https://github.com/python/mypy/issues/5936 for overriding issues.
    @_mock.setter
    def _mock(self, obj: PandasType) -> None:
        # TODO: Maybe we should save args and kwargs from __init__ in
        # order to reproduce them here?
        raise MLIRPandasException(f"cannot set mock to object of type {type(obj)}")

    @property
    def _mlir_builder_safe(self) -> MLIRBuilder:
        """Access _mlir_builder without handling Optional.

        This is for overriden methods, where it is guaranteed by the
        _override decorator that _mlir_builder is not None, but where
        static analysis tools cannot detect that fact.
        """
        if self._mlir_builder is None:
            raise MLIRPandasException("_mlir_builder_safe: _mlir_builder should not be None")
        return self._mlir_builder

    def __delattr__(self, name: str) -> None:
        if name not in dir(self):
            delattr(self._mock, name)
        else:
            super().__delattr__(name)

    def __getattr__(self, name: str) -> Any:
        def wrapper(attr: Any) -> Any:
            def wrapper_call(*args: Any, **kwargs: Any) -> Any:
                result: Any = attr(*args, **kwargs)
                return self._construct_return_value(result)
            return wrapper_call

        attr: Any = self._mock.__getattribute__(name)
        if callable(attr):
            return wrapper(attr)
        return self._construct_return_value(attr)

    def __setattr__(self, name: str, value: Any) -> None:
        if self._FINISHED_INIT in self.__dict__ and name not in dir(self):
            setattr(self._mock, name, value)
        else:
            super().__setattr__(name, value)

    def _aggregate(
        self,
        _ret_type_: Type[Union["DataFrame", "Series"]],
        _group_by_cols_: List[str],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        # Currently only supports kwargs of new column names to
        # Tuples of existing column names to aggregation functions
        # (→ Dict[str, Tuple[str, str]]).
        # See Named Aggregation:
        # https://pandas.pydata.org/docs/user_guide/groupby.html#named-aggregation
        params: Optional[Namespace] = get_params([], args, {})
        aggrfns: Set[str] = set(f.name for f in RelalgAggrFunc)
        if (params is None
                or not isinstance(kwargs, dict)
                or not self._has_col_names(col for col, _ in kwargs.values())
                or any(
                    aggrfn not in aggrfns for _, aggrfn in kwargs.values()
                )):
            raise OverrideException()
        # Create a copy which does not possess any cols.
        ret: Any = self._copy(cols=OrderedSet(), ret_type=DataFrame)
        cols_map: Dict[str, TableColumn] = self._get_cols_map()
        cols_mapping: OrderedDict[str, Tuple[TableColumn, RelalgAggrFunc]] = OrderedDict(
            (new_name, (cols_map[old_name], RelalgAggrFunc[aggrfn]))
            for new_name, (old_name, aggrfn) in kwargs.items()
        )
        group_by_cols: List[TableColumn] = self._get_cols_by_name(_group_by_cols_)
        ret._col_config().update(group_by_cols)
        ret._mlir_builder_safe.aggregate(
            group_by_cols=group_by_cols,
            cols_mapping=cols_mapping
        )

        if group_by_cols:
            if not issubclass(_ret_type_, DataFrame):
                raise MLIRPandasException("groupby should return DataFrame")
            ret = ret.sort_values(
                [tc.name for tc in group_by_cols], ascending=[True] * len(group_by_cols)
            )
            # We have to materialize here because the index got lost during the
            # aggregation.
            ret._execute_mlir(ignore_index=True, table_only=True)
        elif issubclass(_ret_type_, Series):
            # Materialize and set index.
            ret._execute_mlir(ignore_index=True)
            ret = Series(_mock_=ret._mock_.mock.T.iloc[:, 0], _timestamps_=ret._timestamps)
            ret._mock_.mock.name = next(col.name for col, _ in cols_mapping.values())
        elif issubclass(_ret_type_, DataFrame):
            ret._execute_mlir(ignore_index=True)
            aggrfn_names: OrderedSet[str] = OrderedSet(
                aggrfn.name for _, aggrfn in cols_mapping.values()
            )
            col_names: OrderedSet[str] = OrderedSet(
                col.name for col, _ in cols_mapping.values()
            )
            data: Dict[str, Any] = {}
            for col_name in col_names:
                l: List[Any] = []
                for aggrfn_name in aggrfn_names:
                    new_col_name: str = f"__{col_name}_{aggrfn_name}"
                    l.append(
                        ret._mock_.mock[new_col_name].iloc[0]
                        if new_col_name in ret._mock_.mock else None
                    )
            # row order
            ret._mock_ = Data(pd.DataFrame(data))
            ret._mock_.mock.set_index(iter(aggrfn_names), inplace=True)

        return ret

    @overload
    def _col_config(self, mock: None = None) -> OrderedSet[TableColumn]: # type: ignore
        ...

    @overload
    def _col_config(self, mock: Data) -> List[Tuple[str, DtypeObj]]:
        ...

    def _col_config(
        self,
        mock: Optional[Data] = None
    ) -> Union[OrderedSet[TableColumn], List[Tuple[str, DtypeObj]]]:
        """Get the current column configuration.

        Caution! This might change over time, so this should always be
        called when needed. There is no point in storing the column
        configuration.

        Returns a list of tuples containing the names and the types of
        the columns.

        Parameters:
        -----------
        mock    If not None, gives the current column configuration
                in the data object, so *not* considering the not-yet
                materialized operations in self._mlir_builder.
                Defaults to None.
                This parameter has no effect in case _mlir_builder is
                None, so in case that the data cannot be handled
                by the current MLIR support.
        """
        if mock is None and self._mlir_builder is not None:
            return self._mlir_builder.col_config
        elif mock is not None and mock.is_arrow():
            return [
                (n, get_dtype_from_pyarrow_type(t))
                for n, t in zip(mock.mock.schema.names, mock.mock.schema.types)
            ]
        raise NotImplementedError()

    def _copy(
        self,
        copy: bool = False,
        cols: Optional[OrderedSet[TableColumn]] = None,
        ret_type: Optional[Type["MLIRPandas"]] = None,
        ret_table_identifier: Optional[str] = None
    ) -> Any:
        """All copying is done without materialization.

        Parameters:
        -----------
        copy      Create a copy vs create a view.
        cols      Specify for which columns the new object will be
                  responsible. If not given, it will default to the
                  columns of self.

        The difference between a view and a copy is whether the
        two MLIRBuilder objects are independent vs they work on the same
        function and share the same state.
        """
        if ret_type is None:
            ret_type = type(self)
            assert(ret_type != None) # make static analysis happy
        cols = self._col_config().copy() if cols is None else cols.copy()

        table_identifier: str
        if copy:
            if ret_table_identifier is None:
                table_identifier = f"data_{time.time_ns()}"
            else:
                table_identifier = ret_table_identifier
        else:
            table_identifier = self._table_identifier

        mlir_builder: Optional[MLIRBuilder] = None

        if self._mlir_builder is not None:
            # The MLIR builder will create new columns for the new scope if do_copy is True.
            mlir_builder = MLIRBuilder.from_other(
                other=self._mlir_builder,
                table_identifier=table_identifier,
                cols=cols,
                do_copy=copy
            )

        ret: Any = ret_type._init_internal(
            mock=None,
            orig_references=self._get_data(),
            mlir_builder=mlir_builder,
            table_identifier=table_identifier,
            timestamps=self._timestamps
        )

        return ret

    def _execute_mlir(self, ignore_index: bool = False, table_only: bool = False) -> None:
        """Execute the current mlir module and update self._mock_."""
        if self._mlir_builder is None:
            raise MLIRPandasException("_execute_mlir: MLIRBuilder is None")

        self._timestamps.append(("pre load to db", gettime()))
        self._load_data_to_db()
        self._timestamps.append(("post load to db", gettime()))

        cols: OrderedSet[TableColumn] = self._col_config()
        if not ignore_index:
            cols = cols | OrderedSet([self._icol])
        mlirfile: str = self._mlir_builder.materialize(cols)
        logging.debug("########## mlirfile ##########\n%s\n####################", mlirfile)

        self._timestamps.append(("pre run", gettime()))
        table: pa.Table = db_run(mlirfile)
        self._timestamps.append(("post run", gettime()))

        self._timestamps.append(("pre load from db", gettime()))
        if table_only:
            self._mock_ = Data(table)
        else:
            self._load_data_from_db(table, ignore_index)
            del table
        self._timestamps.append(("post load from db", gettime()))
        self._orig_references = {}

    def _get_data(self) -> Dict[str, Data]:
        result: Dict[str, Data] = {}

        if self._mock_ is not None:
            result[self._table_identifier] = self._mock_
        if self._orig_references:
            result.update(self._orig_references)

        if not result:
            raise MLIRPandasException("both mock object and orig_references are empty")
        return result

    def _get_cols_by_name(self, col_names: Iterable[str]) -> List[TableColumn]:
        """Also handle hidden index column."""
        cols_map: OrderedDict[str, TableColumn] = self._get_cols_map()
        return [cols_map[col_name] for col_name in col_names]

    @overload
    def _get_cols_map(self, mock: None = None) -> OrderedDict[str, TableColumn]:
        ...

    @overload
    def _get_cols_map(self, mock: Data) -> OrderedDict[str, DtypeObj]:
        ...

    def _get_cols_map(
        self,
        mock: Optional[Data] = None
    ) -> Union[OrderedDict[str, TableColumn], OrderedDict[str, DtypeObj]]:
        if mock is None:
            return OrderedDict((tc.name, tc) for tc in self._col_config(mock))
        else:
            return OrderedDict((col_name, dtype) for col_name, dtype in self._col_config(mock))

    def _has_col_names(self, col_names: Iterable[str]) -> bool:
        return not (set(col_names) - set(tc.name for tc in self._col_config()))

    def _load_data_to_db(self) -> None:
        to_load: Dict[str, pa.Table] = {}

        for table_identifier, obj in self._get_data().items():
            self._timestamps.append(("pre convert to pa.Table", gettime()))
            table: pa.Table = obj.to_arrow(self._icol.name)
            self._timestamps.append(("post convert to pa.Table", gettime()))
            to_load[table_identifier] = table

        db_load(to_load)

    def _load_data_from_db(
        self,
        table: pa.Table,
        ignore_index: bool = False
    ) -> None:
        obj: Union[pd.DataFrame, pd.Series] = table.to_pandas(**PyArrow2PandasConversionArgs)
        # The pyarrow docs state that the to_pandas method may also return a
        # Series. This seems not to match the actual behaviour.
        if isinstance(obj, pd.Series):
            raise MLIRPandasException("got unexpected Series from pyarrow")
        logging.debug(f"result from db: \n{obj}")
        if not ignore_index:
            # Reset the index using the internal storage of the original index.
            # TODO: can support multi-index.
            obj.set_index(self._icol.name, inplace=True)
            # TODO: store the original index name.
            obj.index.name = None
        self._mock = obj

    def _mlir_builder_check_support(self, _: Any) -> bool:
        raise NotImplementedError()

    def _mlir_builder_create(
        self,
        mock: Data,
        icols: List[TableColumn]
    ) -> Optional[MLIRBuilder]:
        if not mock.is_arrow() and not self._mlir_builder_check_support(mock.mock):
            return None
        try:
            return MLIRBuilder(
                table_identifier=self._table_identifier,
                icols=icols,
                columns=self._col_config(mock=mock)
            )
        except (UnsupportedDataType, UnsupportedColumnName):
            logging.debug("unsupported data, mlir_builder will be None")
            return None

    def _reset_timestamps(self) -> None:
        self._timestamps.append(("reset_timestamps", gettime()))


# cf. https://peps.python.org/pep-3115/#example
class Meta(type):
    do_not_override: Set[str] = set([
        "__class__", "__dict__", "__dir__", "__getattribute__", "__init__", "__init_subclass__",
        "__new__", "__reduce__", "__reduce_ex__", "__subclasshook__"
    ])
    parentclass: Optional[Type[Any]] = None

    def __new__(
        cls,
        name: str,
        bases: Tuple[Type[Any], ...],
        classdict: Dict[str, Any],
        **kwargs: Any
    ) -> Type[Any]:
        def wrapper(attr: str) -> Any:
            def wrapper_call(self: Any) -> Any:
                return result.__getattr__(self, attr) # type: ignore
            return wrapper_call

        def wrapper_callable(attr: str) -> Any:
            def wrapper_call(self: Any, *args: Any, **kwargs: Any) -> Any:
                return result.__getattr__(self, attr)(*args, **kwargs) # type: ignore
            return wrapper_call

        if cls.parentclass is None:
            raise ValueError("cls.parentclass not defined")

        is_subclass: bool = False
        # Use the original class as baseclass when the custom class
        # gets subclassed. (Duplicate base classes are not allowed, but
        # that will be handled elsewhere.)
        while True:
            try:
                base_index: int = tuple(map(type, bases)).index(cls)
                bases = (*bases[:base_index], cls.parentclass, *bases[base_index + 1:])
                is_subclass = True
            except ValueError:
                # The custom class is not in the list of base classes.
                break

        result: Type[Any] = type.__new__(cls, name, bases, classdict, **kwargs)
        if is_subclass:
            return result

        # Fix double underscore attributes which are not handled by
        # __getattr__ by default.
        # Redirect double underscore methods from "object" or
        # the parentclass per default to the implementation of the
        # parentclass if they are not overwritten by the custom class.
        #
        # TODO: dangerous? (yes, that's why there is "do_not_override"... - sufficient?)
        # TODO: static methods?
        protected_attrs: Set[str] = (
            set(result.__dict__.keys())
            | set(MLIRPandas.__dict__.keys())
            | cls.do_not_override
        )
        excl_df_attrs: Set[str] = set(dir(cls.parentclass)) - protected_attrs
        excl_obj_attrs: Set[str] = set(dir(object)) - protected_attrs
        for attr in excl_df_attrs | excl_obj_attrs:
            if not attr.startswith("__"):
                continue
            elif callable(getattr(cls.parentclass, attr)):
                setattr(result, attr, wrapper_callable(attr))
            else:
                setattr(result, attr, wrapper(attr))

        return result


class DataFrameMeta(Meta):
    parentclass = pd.DataFrame


class DataFrameGroupByMeta(Meta):
    parentclass = pd_gb.DataFrameGroupBy


class SeriesMeta(Meta):
    parentclass = pd.Series


class DataFrameGroupBy(MLIRPandas, metaclass=DataFrameGroupByMeta):
    _MOCKCLASS = pd_gb.DataFrameGroupBy

    def _init_custom(self) -> None:
        super()._init_custom()
        self._group_by_cols: List[str]
        if self._mock_ is None:
            self._group_by_cols = []
        elif not isinstance(self._mock_.mock, pd_gb.DataFrameGroupBy):
            raise MLIRPandasException("DataFrameGroupBy: wrong mock type")
        else:
            self._group_by_cols = [str(name) for name in self._mock_.mock.grouper.names]
        self._sort_group_keys: bool = True

    @staticmethod
    def _to_df(obj: pd_gb.DataFrameGroupBy) -> pd.DataFrame:
        return obj.obj

    @MLIRPandas._mock.setter # type: ignore[attr-defined]
    def _mock(self, obj: PandasType) -> None:
        if isinstance(obj, pd_gb.DataFrameGroupBy):
            self._mock_ = Data(obj)
        elif isinstance(obj, pd.DataFrame):
            self._mock_ = Data(pd_gb.DataFrameGroupBy(obj))
        else:
            super()._mock(obj)

    @overload
    def _col_config(self, mock: None = None) -> OrderedSet[TableColumn]: # type: ignore
        ...

    @overload
    def _col_config(self, mock: Data) -> List[Tuple[str, DtypeObj]]:
        ...

    def _col_config(
        self,
        mock: Optional[Data] = None
    ) -> Union[OrderedSet[TableColumn], List[Tuple[str, DtypeObj]]]:
        if mock is not None and not mock.is_arrow():
            return list(zip(mock.mock.obj.columns, mock.mock.obj.dtypes))
        return super()._col_config(mock)

    def _mlir_builder_check_support(self, _: pd_gb.DataFrameGroupBy) -> bool:
        return True

    @override("agg", "DataFrame", True)
    def agg(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self.aggregate(*args, **kwargs)

    @override("aggregate", "DataFrame", True)
    def aggregate(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._aggregate(DataFrame, self._group_by_cols, *args, **kwargs)


class ArrayLike(MLIRPandas):
    def _arith_op(self: ArrayLikeType, other: Optional[Any], op: DBArithOp) -> ArrayLikeType:
        ret: ArrayLikeType = self._copy(cols=OrderedSet()) #) #copy=True, cols=OrderedSet())
        result_name: str = f"__arith_result_{time.time_ns()}"
        if other is None:
            try:
                ret._mlir_builder_safe.arith_cols(
                    op, None, self._col_config(), result_name=result_name
                )
            except:
                raise OverrideException()
        else:
            try:
                ret._mlir_builder_safe.arith_constant(
                    op, other, self._col_config(), result_name=result_name
                )
            except (MLIRBuilderException, SupportedTypesException):
                raise OverrideException()
        return ret

    def _concat(self, other: "ArrayLike", inplace: bool = False) -> "DataFrame":
        """Concatenate the columns of this object with the ones of the other."""
        df: "DataFrame"
        same_func: bool = self._mlir_builder_safe.func is other._mlir_builder_safe.func
        #same_table: bool = self._same_table(self, other)

        if inplace:
            if not isinstance(self, DataFrame):
                raise MLIRPandasException("can only concat inplace to DataFrame")
            df = self
        else:
            df = self._copy(ret_type=DataFrame) #, copy=same_table)

        if same_func:
            df._col_config().update(other._col_config())
        else:
            df = df.merge(
                other,
                how="inner",
                left_on=[],
                right_on=[],
                _index_=True,
                _inplace_=inplace
            )

        return df

    def _head(self: ArrayLikeType, *args: Any, **kwargs: Any) -> ArrayLikeType:
        params: Optional[Namespace] = get_params([], args, kwargs, OrderedDict(n=5))
        if params is None or not isinstance(params.n, int) or params.n >= 2**32:
            raise OverrideException()
        ret: ArrayLikeType = self.copy(copy=True)
        ret._mlir_builder_safe.limit(params.n)
        return ret

    def _sort(self: ArrayLikeType, *args: Any, **kwargs: Any) -> ArrayLikeType:
        params: Optional[Namespace] = get_params(["by"], args, kwargs, OrderedDict(ascending=True))
        if (params is None or (
                not (isinstance(params.by, str) and isinstance(params.ascending, bool))
            and not (
                isinstance(params.by, list)
                and isinstance(params.ascending, list)
                and all(isinstance(by, str) for by in params.by)
                and all(isinstance(asc, bool) for asc in params.ascending)
                and len(params.by) == len(params.ascending)
            ))):
            raise OverrideException()
        by: List[str] = params.by if isinstance(params.by, list) else [params.by]
        if not self._has_col_names(by):
            raise OverrideException()
        ret: ArrayLikeType = self._copy() #copy=True)
        ret._mlir_builder_safe.sort([
            (col, RelalgSortSpec.asc if asc else RelalgSortSpec.desc)
            for col, asc in zip(
                self._get_cols_by_name(by),
                params.ascending if isinstance(params.ascending, list) else [params.ascending]
            )
        ])
        return ret



class Series(ArrayLike, metaclass=SeriesMeta):
    _MOCKCLASS = pd.Series
    # Suppress the warning:
    # FutureWarning: The default dtype for empty Series will be 'object' instead
    # of 'float64' in a future version. Specify a dtype explicitly to silence
    # this warning.
    _DEFAULT_KWARGS = {"dtype": 'object'}

    @staticmethod
    def _to_df(obj: pd.Series, name: Optional[str] = None) -> pd.DataFrame:
        if name is None:
            name = str(obj.name)
            #raise MLIRPandasException("unnamed Series currently not supported")
        return pd.DataFrame({name: obj}, copy=False)

    @MLIRPandas._mock.setter # type: ignore[attr-defined]
    def _mock(self, obj: PandasType) -> None:
        if isinstance(obj, pd.Series):
            self._mock_ = Data(obj)
        elif isinstance(obj, pd.DataFrame):
            # Get the first column of the DataFrame.
            self._mock_ = Data(obj.iloc[:, 0])
        else:
            raise MLIRPandasException(f"cannot set mock to object of type {type(obj)}")

    @override("__add__")
    def __add__(self, other: Any) -> "Series":
        return self._arith_op(other, DBArithOp.add)

    @override("__and__")
    def __and__(self, other: Any) -> "Series":
        return self._arith_op(other, DBArithOp.and_op)

    @override("__eq__")
    def __eq__(self, other: Union[Any, "Series"]) -> "Series":
        return self._cmp(other, DBCmpPredicate.EQ)

    @override("__gt__")
    def __gt__(self, other: Union[Any, "Series"]) -> "Series":
        return self._cmp(other, DBCmpPredicate.GT)

    @override("__ge__")
    def __ge__(self, other: Union[Any, "Series"]) -> "Series":
        return self._cmp(other, DBCmpPredicate.GTE)

    @override("__invert__")
    def __invert__(self) -> "Series":
        return self._arith_op(None, DBArithOp.not_op)

    @override("__lt__")
    def __lt__(self, other: Union[Any, "Series"]) -> "Series":
        return self._cmp(other, DBCmpPredicate.LT)

    @override("__le__")
    def __le__(self, other: Union[Any, "Series"]) -> "Series":
        return self._cmp(other, DBCmpPredicate.LTE)

    @override("__mod__")
    def __mod__(self, other: Any) -> "Series":
        return self._arith_op(other, DBArithOp.mod)

    @override("__mul__")
    def __mul__(self, other: Any) -> "Series":
        return self._arith_op(other, DBArithOp.mul)

    @override("__ne__")
    def __ne__(self, other: Union[Any, "Series"]) -> "Series":
        return self._cmp(other, DBCmpPredicate.NEQ)

    @override("__or__")
    def __or__(self, other: Any) -> "Series":
        return self._arith_op(other, DBArithOp.or_op)

    @override("__sub__")
    def __sub__(self, other: Any) -> "Series":
        return self._arith_op(other, DBArithOp.sub)

    @override("__truediv__")
    def __truediv__(self, other: Any) -> "Series":
        return self._arith_op(other, DBArithOp.div)

    def _arith_op(self, other: Optional[Any], op: DBArithOp) -> "Series":
        if not isinstance(other, Series) or other._mlir_builder is None:
            return super()._arith_op(other, op)
        df: "DataFrame" = self._concat(other)
        result_name: str = f"__arith_result_{time.time_ns()}"
        df._mlir_builder_safe.arith_cols(
            op=op,
            col=next(iter(other._col_config())),
            columns=self._col_config(),
            result_name=result_name
        )
        return df._copy(
            #copy=True,
            cols=OrderedSet(col for col in df._col_config() if col.name == result_name),
            ret_type=Series
        )

    def _cmp(self, other: Union[Any, "Series"], predicate: DBCmpPredicate) -> "Series":
        ret: "Series"
        new_column: str = f"__cmp_{time.time_ns()}"

        if isinstance(other, Series) and other._mlir_builder is not None:
            df: "DataFrame" = self._concat(other)
            df._mlir_builder_safe.compare_cols(
                left=next(iter(self._col_config())),
                right=next(iter(other._col_config())),
                op=predicate,
                new_column=new_column,
                new_scope=df._table_identifier
            )
            ret = df._copy(
                #copy=True,
                cols=OrderedSet([df._mlir_builder_safe.table_context.cols.popitem()]),
                ret_type=Series
            )
        else:
            ret = self._copy(cols=OrderedSet()) #copy=True, cols=OrderedSet())
            # Add a new column with the scope of the new Series and the
            # True/False values as content.
            tc: TableColumn = next(iter(self._col_config()))
            try:
                ret._mlir_builder_safe.compare_constant(
                    column=tc,
                    constant=other,
                    op=predicate,
                    new_column=new_column,
                    new_scope=ret._table_identifier
                )
            except InvalidArgumentException:
                raise OverrideException()
        return ret

    @overload
    def _col_config(self, mock: None = None) -> OrderedSet[TableColumn]: # type: ignore
        ...

    @overload
    def _col_config(self, mock: Data) -> List[Tuple[str, DtypeObj]]:
        ...

    def _col_config(
        self,
        mock: Optional[Data] = None
    ) -> Union[OrderedSet[TableColumn], List[Tuple[str, DtypeObj]]]:
        if mock is not None and not mock.is_arrow():
            # If the series has no name, the name will be 'None'.
            return [(str(mock.mock.name), mock.mock.dtype)]
        return super()._col_config(mock)

    def _mlir_builder_check_support(self, _: pd.Series) -> bool:
        """Require a string as name."""
        return True #isinstance(mock.name, str)

    @override("agg")
    def agg(self, *args: Any, **kwargs: Any) -> Any:
        return self.aggregate(*args, **kwargs)

    @override("aggregate")
    def aggregate(self, *args: Any, **kwargs: Any) -> Any:
        col_name: str = next(iter(self._col_config())).name
        kw_args: Dict[str, Any] = {
            new_col_name: (col_name, func)
            for new_col_name, func in kwargs.items()
        }
        return self._aggregate(Series, [], *args, **kw_args)

    @override("head")
    def head(self, *args: Any, **kwargs: Any) -> "Series":
        return self._head(*args, **kwargs)

    @override("sort_values")
    def sort_values(self, *args: Any, **kwargs: Any) -> "Series":
        return self._sort(by=next(iter(self._col_config())), *args, **kwargs)


class DataFrame(ArrayLike, metaclass=DataFrameMeta):
    _MOCKCLASS = pd.DataFrame

    @staticmethod
    def _to_df(obj: pd.DataFrame) -> pd.DataFrame:
        return obj

    @MLIRPandas._mock.setter # type: ignore[attr-defined]
    def _mock(self, obj: PandasType) -> None:
        if isinstance(obj, pd.DataFrame):
            self._mock_ = Data(obj)
        elif isinstance(obj, pd.Series):
            self._mock_ = Data(pd.DataFrame({obj.name: obj}, copy=False))
        else:
            raise MLIRPandasException(f"cannot set mock to object of type {type(obj)}")

    @override("__add__")
    def __add__(self, other: Any) -> "DataFrame":
        return self._arith_op(other, DBArithOp.add)

    @override("__getitem__")
    def __getitem__(self, key: Union[List[Any], Any]) -> Union["Series", "DataFrame"]:
        """Support slicing.

        See https://pandas.pydata.org/docs/user_guide/indexing.html#basics.

        Note: The original implementation does not define when slicing
        results in a copy or a view! In this implementation, we do not
        copy.
        """
        ret: Union["Series", "DataFrame"]

        if isinstance(key, pd.Series):
            key = Series(_mock_=key)

        if isinstance(key, Series):
            ret = self._concat(key)
            sel_col: TableColumn = next(iter(key._col_config()))
            ret._mlir_builder_safe.selection_col(True, sel_col)
            # Remove the selection column again. TODO: necessary?
            #ret._mlir_builder_safe.projection(self._col_config())
            # Remove the selection column only from the col config.
            ret._col_config().difference_update([sel_col])
        else:
            keys: List[str] = key if isinstance(key, list) else [key]
            if not self._has_col_names(keys):
                raise OverrideException()
            ret = self._copy(
                copy=False,
                cols=OrderedSet(tc for tc in self._col_config() if tc.name in keys),
                ret_type=DataFrame if isinstance(key, list) else Series
            )
        return ret

    @override("__mod__")
    def __mod__(self, other: Any) -> "DataFrame":
        return self._arith_op(other, DBArithOp.mod)

    @override("__mul__")
    def __mul__(self, other: Any) -> "DataFrame":
        return self._arith_op(other, DBArithOp.mul)

    @override("__setitem__")
    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(key, str):
            raise OverrideException()
        other: "Series"
        if isinstance(value, pd.Series):
            other = Series(_mock_=value)
        elif isinstance(value, Series):
            other = value
        else:
            raise OverrideException()

        self._concat(other, inplace=True)
        self._mlir_builder_safe.rename(OrderedDict(
            (tc, TableColumn(self._table_identifier, key, tc.type))
            for tc in other._col_config()
        ))

    @override("__sub__")
    def __sub__(self, other: Any) -> "DataFrame":
        return self._arith_op(other, DBArithOp.sub)

    @override("__truediv__")
    def __truediv__(self, other: Any) -> "DataFrame":
        # Caution: In order to be consistent with Python, the DB should
        # not produce integer results.
        if isinstance(other, int):
            other = float(other)
        return self._arith_op(other, DBArithOp.div)

    @overload
    def _col_config(self, mock: None = None) -> OrderedSet[TableColumn]: # type: ignore
        ...

    @overload
    def _col_config(self, mock: Data) -> List[Tuple[str, DtypeObj]]:
        ...

    def _col_config(
        self,
        mock: Optional[Data] = None
    ) -> Union[OrderedSet[TableColumn], List[Tuple[str, DtypeObj]]]:
        if mock is not None and not mock.is_arrow():
            return list(zip(mock.mock.columns, mock.mock.dtypes))
        return super()._col_config(mock)

    def _mlir_builder_check_support(self, mock: pd.DataFrame) -> bool:
        """Currently no support for named index or column labels.

        Additionally, all column labels need to be strings.
        """
        return (
            mock.columns.name is None
            and mock.index.name is None
            and all(isinstance(col_name, str) for col_name in mock.columns)
        )

    @override("agg", "DataFrame", True)
    def agg(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self.aggregate(*args, **kwargs)

    @override("aggregate", "DataFrame", True)
    def aggregate(self, *args: Any, **kwargs: Any) -> "DataFrame":
        params: Optional[Namespace] = get_params(["func"], args, kwargs)
        if (params is None
                or not isinstance(params.func, dict)
                or any(not isinstance(k, list) for k in params.func.values())):
            raise OverrideException()
        kw_args: Dict[str, Any] = {
            f"__{col}_{func}": (col, func)
            for col, funcs in params.func
            for func in funcs
        }
        return self._aggregate(DataFrame, [], *args, **kw_args)

    @override("copy")
    def copy(self, deep: bool = True) -> "DataFrame":
        # TODO: There might be some problems here. See the notes in:
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html
        return super()._copy(copy=deep)

    @override("drop_duplicates")
    def drop_duplicates(self, *args: Any, **kwargs: Any) -> "DataFrame":
        params: Optional[Namespace] = get_params([], args, kwargs)
        if params is None:
            raise OverrideException()
        ret: "DataFrame" = self._copy()
        ret._mlir_builder_safe.drop_duplicates()
        return ret

    @override("dropna")
    def dropna(self, *args: Any, **kwargs: Any) -> "DataFrame":
        params: Optional[Namespace] = get_params(["subset"], args, kwargs)
        if (params is None
                or not isinstance(params.subset, str)
                or not (
                    isinstance(params.subset, list)
                    and all(isinstance(s, str) for s in params.subset)
                )):
            raise OverrideException()
        subset: List[str] = params.subset if isinstance(params.subset, list) else [params.subset]
        if not self._has_col_names(subset):
            raise OverrideException()
        ret: "DataFrame" = self._copy()
        ret._mlir_builder_safe.dropna(self._get_cols_by_name(subset))
        return ret

    @override("groupby", "DataFrameGroupBy", True)
    def groupby(self, *args: Any, **kwargs: Any) -> DataFrameGroupBy:
        # TODO: sort, dropna
        params: Optional[Namespace] = get_params(
            ["by", "as_index"], args, kwargs, OrderedDict(sort=True)
        )
        if params is None or not isinstance(params.sort, bool):
            raise OverrideException()
        by: List[Any] = params.by if isinstance(params.by, list) else [params.by]
        if params.as_index or not self._has_col_names(by):
            raise OverrideException()
        ret: DataFrameGroupBy = self._copy(ret_type=DataFrameGroupBy) #copy=True
        ret._group_by_cols = by
        ret._sort_group_keys = params.sort
        return ret

    @override("head")
    def head(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._head(*args, **kwargs)

    @override("merge")
    def merge(
        self,
        *args: Any,
        _inplace_: bool = False,
        _index_: bool = False,
        **kwargs: Any
    ) -> "DataFrame":
        # TODO: "validate" parameter (see docs)?
        # Re "suffixes": renaming can be done seperately.
        # TODO:
        # "If both key columns contain rows where the key is a null value, those rows
        # will be matched against each other. This is different from usual SQL join
        # behaviour and can lead to unexpected results."
        # (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)
        kw_args: Dict[str, Any] = kwargs.copy()
        kw_args.update({"_inplace_": _inplace_, "_index_": _index_})
        params: Optional[Namespace] = get_params(
            wanted=["right", "how", "left_on", "right_on", "_inplace_", "_index_"],
            args=args,
            kwargs=kw_args,
        )
        if params is None:
            raise OverrideException()
        right: "DataFrame"
        if isinstance(params.right, DataFrame) or isinstance(params.right, Series):
            right = params.right._copy(ret_type=DataFrame)
            if right._mlir_builder is None:
                raise OverrideException()
            right._mlir_builder_safe._detach_func()
        elif isinstance(params.right, pd.DataFrame):
            right = DataFrame(_mock_=params.right)
        elif isinstance(params.right, pd.Series):
            right = DataFrame(_mock_=Series._to_df(params.right))
        else:
            raise OverrideException()

        left_on: OrderedSet[str] = OrderedSet(
            params.left_on if isinstance(params.left_on, list) else [params.left_on]
        )
        right_on: OrderedSet[str] = OrderedSet(
            params.right_on if isinstance(params.right_on, list) else [params.right_on]
        )
        if (not self._has_col_names(left_on)
                or not right._has_col_names(right_on)
                or right._mlir_builder is None
                or params.how not in MergeFlavor.__members__):
            raise OverrideException()

        left_obj: "DataFrame" = self if params.how != "right" else right
        right_obj: "DataFrame" = right if params.how != "right" else self
        cols: OrderedSet[TableColumn] = left_obj._col_config() | right_obj._col_config()

        ret: "DataFrame"
        if _inplace_:
            ret = left_obj
            ret._orig_references.update(right_obj._get_data())
            ret._mlir_builder = MLIRBuilder.combine(
                left=ret._mlir_builder_safe,
                right=right_obj._mlir_builder_safe,
                table_identifier=ret._table_identifier,
                table_context_cols=ret._col_config() | right_obj._col_config()
            )
        else:
            orig_references: Dict[str, Data] = left_obj._get_data()
            orig_references.update(right_obj._get_data())
            ret = DataFrame._init_internal(
                mock=None,
                orig_references=orig_references,
                mlir_builder=left_obj._mlir_builder,
                other_mlir_builder=right_obj._mlir_builder,
                table_context_cols=cols,
                timestamps=sorted(right_obj._timestamps + left_obj._timestamps, key=lambda t: t[1])
            )

        ret._mlir_builder_safe.merge(
            merge_flavor=MergeFlavor[params.how],
            left_on=[left_obj._icol] if _index_ else ret._get_cols_by_name(left_on),
            right_on=[right_obj._icol] if _index_ else ret._get_cols_by_name(right_on)
        )

        return ret

    @override("rename")
    def rename(self, *args: Any, **kwargs: Any) -> "DataFrame":
        # TODO: inplace untested, maybe not needed
        params: Optional[Namespace] = get_params(
            ["columns"], args, kwargs, optional=OrderedDict(inplace=False)
        )
        if (params is None
                or not isinstance(params.columns, dict)
                or not all(isinstance(k, str) and isinstance(v, str)
                           for k, v in params.columns.items())
                or not self._has_col_names(params.columns.keys())):
            raise OverrideException()
        ret: "DataFrame"
        ret = self if params.inplace else self._copy(copy=True)
        ret._mlir_builder_safe.rename(OrderedDict(
            (tc, TableColumn(tc.scope, params.columns[tc.name], tc.type))
            for tc in ret._col_config()
            if tc.name in params.columns
        ))
        return ret

    @override("sort_values")
    def sort_values(self, *args: Any, **kwargs: Any) -> "DataFrame":
        return self._sort(*args, **kwargs)


def get_params(
    wanted: List[str],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    optional: OrderedDict[str, Any] = OrderedDict()
) -> Optional[Namespace]:
    result: Namespace = Namespace()
    args_i: int = 0
    for arg, opt in itertools.chain(((w, False) for w in wanted), ((o, True) for o in optional)):
        if kwargs and arg in kwargs:
            setattr(result, arg, (kwargs[arg]))
        elif args and args_i < len(args):
            setattr(result, arg, (args[args_i]))
            args_i += 1
        elif opt:
            setattr(result, arg, optional[arg])
        else:
            return None
    # Check whether there are more arguments.
    if (args and args_i < len(args)) or len(kwargs) > (len(wanted) + len(optional)) - args_i:
        return None
    return result


_CLASSES: Dict[str, Type[MLIRPandas]] = {
    "DataFrame": DataFrame,
    "DataFrameGroupBy": DataFrameGroupBy,
    "Series": Series
}
_MOCK_MAP: Dict[Type[PandasType], Type[MLIRPandas]] = {
    pd.DataFrame: DataFrame,
    pd_gb.DataFrameGroupBy: DataFrameGroupBy,
    pd.Series: Series
}
