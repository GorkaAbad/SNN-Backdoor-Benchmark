"""Generic name-to-class registry."""

from collections.abc import Callable
from typing import Generic, TypeVar

from sbb.utils.errors import SbbError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps registry names to plugins (classes or builder functions).

    Args:
        kind: What the registry holds, used in error messages.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers its argument under name."""

        def decorator(item: T) -> T:
            if name in self._items:
                raise SbbError(f"duplicate {self.kind} name {name!r}")
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        """Return the plugin registered under name."""
        try:
            return self._items[name]
        except KeyError:
            raise SbbError(f"unknown {self.kind} {name!r}, known: {self.names()}") from None

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._items)
