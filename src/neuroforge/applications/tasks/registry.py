"""Central catalog of class-based training tasks.

A single source of truth for *what tasks exist*, so dispatchers (the CLI,
the dashboard server) can look a task up by key instead of hard-coding an
if/elif chain. Adding a new class-based task is a one-line entry here.

Config and task classes are imported **lazily** (inside the loader) so that
importing this module stays cheap — the CLI can resolve ``--help`` and
validate task names without pulling in torch.

The vision-classification path is intentionally *not* listed here: it is
driven by a runner function (`
euroforge.interfaces.cli.commands.vision``) plus extra
monitor wiring rather than a plain ``(config, task)`` class pair, so it is
dispatched separately by its caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["TASK_REGISTRY", "TaskSpec", "get_task_spec"]


@dataclass(frozen=True)
class TaskSpec:
    """Metadata describing one runnable, class-based task."""

    key: str
    label: str
    _load: Callable[[], tuple[type, type]]

    def load(self) -> tuple[type, type]:
        """Return ``(config_cls, task_cls)``, importing them on first use."""
        return self._load()


def _load_logic_gate() -> tuple[type, type]:
    from neuroforge.applications.tasks.logic_gates import LogicGateConfig, LogicGateTask

    return LogicGateConfig, LogicGateTask


def _load_multi_gate() -> tuple[type, type]:
    from neuroforge.applications.tasks.multi_gate import MultiGateConfig, MultiGateTask

    return MultiGateConfig, MultiGateTask


def _load_game_training() -> tuple[type, type]:
    from neuroforge.applications.tasks.game_training import GameTrainingConfig, GameTrainingTask

    return GameTrainingConfig, GameTrainingTask


def _load_evolution() -> tuple[type, type]:
    from neuroforge.applications.tasks.evolution import EvolutionConfig, EvolutionTask

    return EvolutionConfig, EvolutionTask


TASK_REGISTRY: dict[str, TaskSpec] = {
    "evolution": TaskSpec("evolution", "Policy neuroevolution", _load_evolution),
    "logic_gate": TaskSpec("logic_gate", "Single logic gate", _load_logic_gate),
    "multi_gate": TaskSpec("multi_gate", "Multi-gate shared brain", _load_multi_gate),
    "game_training": TaskSpec(
        "game_training", "Vision SNN game playing (online R-STDP)", _load_game_training,
    ),
}


def get_task_spec(key: str) -> TaskSpec | None:
    """Return the :class:`TaskSpec` for *key*, or ``None`` if unknown."""
    return TASK_REGISTRY.get(key)
