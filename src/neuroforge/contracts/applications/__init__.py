"""Application-facing contracts."""

from neuroforge.contracts.applications.evolution import (
    FitnessResult,
    IFitnessEvaluator,
    IGenome,
    IReproduction,
    ISpeciation,
)
from neuroforge.contracts.applications.games import (
    NINTENDO_BUTTONS,
    ControllerAction,
    EpisodeDecision,
    GameClientStep,
    GameObservation,
    GameTransition,
    IEpisodeManager,
    IFrameMetricExtractor,
    IGameClient,
    IRewardModel,
    IVisionGamePolicy,
    ScreenFrame,
    VisionGameMetrics,
)
from neuroforge.contracts.applications.tasks import ILoss, IReadout, ReadoutResult

__all__ = [
    "ControllerAction",
    "EpisodeDecision",
    "FitnessResult",
    "GameClientStep",
    "GameObservation",
    "GameTransition",
    "IEpisodeManager",
    "IFitnessEvaluator",
    "IFrameMetricExtractor",
    "IGameClient",
    "IGenome",
    "ILoss",
    "IReadout",
    "IReproduction",
    "IRewardModel",
    "ISpeciation",
    "IVisionGamePolicy",
    "NINTENDO_BUTTONS",
    "ReadoutResult",
    "ScreenFrame",
    "VisionGameMetrics",
]
