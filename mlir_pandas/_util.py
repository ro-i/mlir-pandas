import itertools
import time
from collections import OrderedDict as OrigOrderedDict
from typing import Container, Generic, Iterable, Optional, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class Decimal:
    precision: int
    scale: int


class OrderedDict(OrigOrderedDict[K, V], Generic[K, V]):
    pass


class OrderedSet(OrigOrderedDict[K, None], Generic[K]):
    def __init__(self, iterable: Optional[Iterable[K]] = None) -> None:
        if iterable is None:
            super().__init__()
        else:
            super().__init__((key, None) for key in iterable)

    def __and__(self, container: Container[K]) -> "OrderedSet[K]":
        return self.intersection(container)

    def __or__(self, other: "OrderedSet[K]") -> "OrderedSet[K]": # type: ignore
        return type(self)(itertools.chain(self, other))

    def __sub__(self, other: "OrderedSet[K]") -> "OrderedSet[K]":
        return self.difference(other)

    def add(self, key: K) -> None:
        self.update([key])

    def difference(self, container: Container[K]) -> "OrderedSet[K]":
        return type(self)(key for key in self if key not in container)

    def difference_update(self, iterable: Iterable[K]) -> None:
        for key in iterable:
            self.pop(key, None)

    def intersection(self, container: Container[K]) -> "OrderedSet[K]":
        return type(self)(key for key in self if key in container)

    def popitem(self, last: bool = True) -> K: # type: ignore
        return super().popitem(last)[0]

    def remove(self, key: K) -> None:
        super().pop(key, None)

    def update(self, iterable: Iterable[K]) -> None: # type: ignore
        super().update([(key, None) for key in iterable])


def gettime() -> int:
    return time.monotonic_ns()
