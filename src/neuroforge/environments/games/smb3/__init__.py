"""Game-playing helpers for vision-only emulator integration."""

from neuroforge.environments.games.clients import (
    ActionProgressGameClient,
    BizHawkClient,
    BizHawkClientConfig,
    ReplayGameClient,
    ScriptedGameClient,
)
from neuroforge.environments.games.smb3.actions import (
    ActionEnergyConfig,
    ActionEnergyModel,
    ActionEnergyStep,
)
from neuroforge.environments.games.smb3.curriculum import ICurriculum, SMB3Curriculum
from neuroforge.environments.games.smb3.environment import VisionOnlyGameLoop
from neuroforge.environments.games.smb3.episode import SMB3EpisodeConfig, SMB3EpisodeManager
from neuroforge.environments.games.smb3.observations import SMB3ObservationSummary
from neuroforge.environments.games.smb3.rewards import (
    SMB3RewardConfig,
    SMB3RewardModel,
    VisionMetricRewardConfig,
    VisionMetricRewardModel,
)
from neuroforge.environments.games.smb3.scoring import SMB3ScoreDelta, score_delta
from neuroforge.environments.games.smb3.termination import (
    SMB3TerminationDecision,
    level_clear_decision,
)

__all__ = [
    "ActionEnergyConfig",
    "ActionEnergyModel",
    "ActionEnergyStep",
    "ActionProgressGameClient",
    "BizHawkClient",
    "BizHawkClientConfig",
    "ICurriculum",
    "ReplayGameClient",
    "SMB3Curriculum",
    "SMB3EpisodeConfig",
    "SMB3EpisodeManager",
    "SMB3ObservationSummary",
    "SMB3RewardConfig",
    "SMB3RewardModel",
    "SMB3ScoreDelta",
    "SMB3TerminationDecision",
    "ScriptedGameClient",
    "VisionMetricRewardConfig",
    "VisionMetricRewardModel",
    "VisionOnlyGameLoop",
    "level_clear_decision",
    "score_delta",
]
