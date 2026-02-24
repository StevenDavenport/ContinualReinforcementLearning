from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(RuntimeError):
    """Raised on registry contract violations."""


@dataclass
class Registry(Generic[T]):
    name: str
    _factories: dict[str, Callable[..., T]] = field(default_factory=dict)

    def register(self, key: str, factory: Callable[..., T], *, replace: bool = False) -> None:
        if not key.strip():
            raise RegistryError("Registry keys must be non-empty.")
        if key in self._factories and not replace:
            raise RegistryError(f"{self.name} registry already contains key '{key}'.")
        self._factories[key] = factory

    def create(self, key: str, *args: object, **kwargs: object) -> T:
        factory = self._factories.get(key)
        if factory is None:
            known = ", ".join(self.names()) or "<empty>"
            raise RegistryError(f"Unknown {self.name} key '{key}'. Known keys: {known}.")
        return factory(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
