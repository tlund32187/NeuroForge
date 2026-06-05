"""Application task factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neuroforge.applications.tasks.registry import TASK_REGISTRY, TaskSpec

__all__ = ["TaskFactory", "build_task_factory"]


def _empty_task_map() -> dict[str, TaskSpec]:
    return {}


@dataclass
class TaskFactory:
    """Factory for class-based application tasks."""

    _tasks: dict[str, TaskSpec] = field(default_factory=_empty_task_map)

    def register(self, spec: TaskSpec) -> None:
        """Register a task spec."""
        if spec.key in self._tasks:
            msg = f"task already registered: {spec.key!r}"
            raise ValueError(msg)
        self._tasks[spec.key] = spec

    def get(self, key: str) -> TaskSpec | None:
        """Return a task spec by key."""
        return self._tasks.get(key)

    def list_keys(self) -> list[str]:
        """Return registered task keys."""
        return sorted(self._tasks)

    def create(self, key: str, **config_kwargs: Any) -> Any:
        """Create a task instance with its config."""
        spec = self._tasks.get(key)
        if spec is None:
            msg = f"unknown task: {key!r}"
            raise KeyError(msg)
        config_cls, task_cls = spec.load()
        return task_cls(config_cls(**config_kwargs))


def build_task_factory() -> TaskFactory:
    """Build a task factory from the application task catalog."""
    factory = TaskFactory()
    for spec in TASK_REGISTRY.values():
        factory.register(spec)
    return factory
