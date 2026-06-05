"""Scripted policy placeholder."""

from __future__ import annotations

from neuroforge.agents.policies.random_policy import RandomPolicy

__all__ = ["ScriptedPolicy"]


class ScriptedPolicy(RandomPolicy):
    """Policy backed by predefined action choices."""
