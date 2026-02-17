"""Learning rule registry."""

from __future__ import annotations

from neuroforge.core.registry import Registry

__all__ = ["LEARNING_RULES", "create_learning_rule"]

LEARNING_RULES: Registry = Registry("learning_rules")


def _register_builtins() -> None:
    """Register built-in learning rules."""
    from neuroforge.learning.rstdp import RSTDPRule

    LEARNING_RULES.register("rstdp", RSTDPRule)


_register_builtins()


def create_learning_rule(key: str, **kwargs: object) -> object:
    """Create a learning rule by registry key."""
    return LEARNING_RULES.create(key, **kwargs)
