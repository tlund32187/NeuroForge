"""Tests for Registry[T] — register, create, aliases, errors."""

from __future__ import annotations

import pytest

from neuroforge.core.registry import Registry


class _DummyA:
    def __init__(self, x: int = 0) -> None:
        self.x = x


class _DummyB:
    def __init__(self, y: str = "hello") -> None:
        self.y = y


@pytest.mark.unit
class TestRegistryBasics:
    def test_register_and_create(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA)
        obj = reg.create("a", x=42)
        assert isinstance(obj, _DummyA)
        assert obj.x == 42

    def test_list_keys_returns_primary_only(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA)
        reg.register("b", _DummyB, aliases=("b_alt",))
        assert reg.list_keys() == ["a", "b"]

    def test_has_primary_key(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA)
        assert reg.has("a") is True
        assert reg.has("missing") is False

    def test_has_alias(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA, aliases=("alpha",))
        assert reg.has("alpha") is True

    def test_create_via_alias(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA, aliases=("alpha",))
        obj = reg.create("alpha", x=7)
        assert isinstance(obj, _DummyA)
        assert obj.x == 7

    def test_contains_operator(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA)
        assert "a" in reg
        assert "missing" not in reg

    def test_len(self) -> None:
        reg: Registry = Registry("test")
        assert len(reg) == 0
        reg.register("a", _DummyA)
        assert len(reg) == 1
        reg.register("b", _DummyB)
        assert len(reg) == 2

    def test_repr(self) -> None:
        reg: Registry = Registry("neurons")
        reg.register("lif", _DummyA)
        r = repr(reg)
        assert "neurons" in r
        assert "lif" in r


@pytest.mark.unit
class TestRegistryErrors:
    def test_duplicate_key_raises(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("a", _DummyB)

    def test_duplicate_alias_raises(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA, aliases=("alpha",))
        with pytest.raises(ValueError, match="already registered"):
            reg.register("b", _DummyB, aliases=("alpha",))

    def test_alias_conflicts_with_primary_raises(self) -> None:
        reg: Registry = Registry("test")
        reg.register("a", _DummyA)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("b", _DummyB, aliases=("a",))

    def test_create_unknown_key_raises(self) -> None:
        reg: Registry = Registry("test")
        with pytest.raises(KeyError, match="Unknown key"):
            reg.create("missing")

    def test_create_unknown_key_shows_available(self) -> None:
        reg: Registry = Registry("test")
        reg.register("lif", _DummyA)
        with pytest.raises(KeyError, match="lif"):
            reg.create("missing")
