"""Generic typed registry — concrete implementation of IRegistry[T].

Maps string keys to constructors.  Supports aliases and duplicate
detection.  No torch dependency.
"""

from __future__ import annotations

from typing import TypeVar

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(list[None]):
    """A named registry mapping string keys to constructors.

    This is a concrete generic registry.  Due to Python's runtime
    limitations with generic classes, we use a simple class with
    explicit typing in method signatures.

    Example
    -------
    >>> reg: Registry[INeuronModel] = Registry("neurons")
    >>> reg.register("lif", LIFModel)
    >>> model = reg.create("lif", tau_mem=20e-3)
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._constructors: dict[str, type[object]] = {}
        self._aliases: dict[str, str] = {}
        self._primary_keys: list[str] = []

    @property
    def name(self) -> str:
        """Registry name (for error messages)."""
        return self._name

    def register(
        self,
        key: str,
        constructor: type[object],
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        """Register a constructor under *key*, with optional aliases.

        Parameters
        ----------
        key:
            Primary registration key.
        constructor:
            The class that produces the registered type.
        aliases:
            Alternative names that also resolve to *constructor*.

        Raises
        ------
        ValueError:
            If *key* or any alias is already registered.
        """
        if key in self._constructors or key in self._aliases:
            msg = f"[{self._name}] Key already registered: {key!r}"
            raise ValueError(msg)

        for alias in aliases:
            if alias in self._constructors or alias in self._aliases:
                msg = f"[{self._name}] Alias already registered: {alias!r}"
                raise ValueError(msg)

        self._constructors[key] = constructor
        self._primary_keys.append(key)

        for alias in aliases:
            self._aliases[alias] = key
            self._constructors[alias] = constructor

    def create(self, key: str, **kwargs: object) -> object:
        """Instantiate from the constructor registered under *key*.

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
        constructor = self._constructors.get(key)
        if constructor is None:
            msg = (
                f"[{self._name}] Unknown key: {key!r}. Available: {self._primary_keys}"
            )
            raise KeyError(msg)
        return constructor(**kwargs)

    def list_keys(self) -> list[str]:
        """Return all registered primary keys (not aliases)."""
        return list(self._primary_keys)

    def has(self, key: str) -> bool:
        """Return True if *key* (or an alias) is registered."""
        return key in self._constructors

    def __contains__(self, key: object) -> bool:
        """Support ``key in registry``."""
        if not isinstance(key, str):
            return False
        return self.has(key)

    def __len__(self) -> int:
        """Number of primary registrations."""
        return len(self._primary_keys)

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, keys={self._primary_keys})"
