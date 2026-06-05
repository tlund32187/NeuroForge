"""NeuroForge task runners."""

from neuroforge.applications.tasks.evolution import EvolutionConfig, EvolutionResult, EvolutionTask
from neuroforge.applications.tasks.logic_gates import (
    GATE_TABLES,
    LogicGateConfig,
    LogicGateResult,
    LogicGateTask,
)
from neuroforge.applications.tasks.multi_gate import (
    ALL_GATES,
    GATE_INDEX,
    MultiGateConfig,
    MultiGateResult,
    MultiGateTask,
)
from neuroforge.applications.tasks.vision_classification import (
    SyntheticVisionBatch,
    SyntheticVisionDataLoader,
    VisionClassificationConfig,
    VisionClassificationResult,
    VisionClassificationTask,
)

__all__ = [
    "ALL_GATES",
    "EvolutionConfig",
    "EvolutionResult",
    "EvolutionTask",
    "GATE_INDEX",
    "GATE_TABLES",
    "LogicGateConfig",
    "LogicGateResult",
    "LogicGateTask",
    "MultiGateConfig",
    "MultiGateResult",
    "MultiGateTask",
    "SyntheticVisionBatch",
    "SyntheticVisionDataLoader",
    "VisionClassificationConfig",
    "VisionClassificationResult",
    "VisionClassificationTask",
]
