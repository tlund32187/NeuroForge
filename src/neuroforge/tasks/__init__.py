"""NeuroForge task runners."""

from neuroforge.tasks.logic_gates import (
    GATE_TABLES,
    LogicGateConfig,
    LogicGateResult,
    LogicGateTask,
)
from neuroforge.tasks.multi_gate import (
    ALL_GATES,
    GATE_INDEX,
    MultiGateConfig,
    MultiGateResult,
    MultiGateTask,
)

__all__ = [
    "ALL_GATES",
    "GATE_INDEX",
    "GATE_TABLES",
    "LogicGateConfig",
    "LogicGateResult",
    "LogicGateTask",
    "MultiGateConfig",
    "MultiGateResult",
    "MultiGateTask",
]
