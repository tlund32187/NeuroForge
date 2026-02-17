"""Registry / factory protocol.

Provides a generic typed registry that maps string keys to constructors,
following the Open/Closed Principle — new implementations are registered
without modifying existing code.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["IRegistry"]

T = TypeVar("T")


@runtime_checkable
class IRegistry(Protocol[T]):
    """A named registry mapping string keys to constructors.

    Implementations should support registration, lookup, and enumeration
    of available keys.
    """

    def register(
        self,
        key: str,
        constructor: type[T],
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Register a constructor under *key*, with optional aliases.

        Parameters
        ----------
        key:
            Primary registration key (e.g. ``"lif"``).
        constructor:
            The class or callable that produces ``T``.
        aliases:
            Alternative names that also resolve to *constructor*.

        Raises
        ------
        ValueError:
            If *key* or any alias is already registered.
        """
        ...

    def create(self, key: str, **kwargs: object) -> T:
        """Instantiate ``T`` from the constructor registered under *key*.

        Parameters
        ----------
        key:
            Registration key or alias.
        **kwargs:
            Forwarded to the constructor.

        Raises
        ------
        KeyError:
            If *key* is not registered.
        """
        ...

    def list_keys(self) -> list[str]:
        """Return all registered primary keys (not aliases)."""
        ...

    def has(self, key: str) -> bool:
        """Return True if *key* (or an alias) is registered."""
        ...
