"""Game-playing helpers for vision-only emulator integration."""

from neuroforge.game.action_energy import ActionEnergyConfig, ActionEnergyModel, ActionEnergyStep
from neuroforge.game.clients import (
    ActionProgressGameClient,
    BizHawkClient,
    BizHawkClientConfig,
    ReplayGameClient,
    ScriptedGameClient,
)
from neuroforge.game.curriculum import ICurriculum, SMB3Curriculum
from neuroforge.game.episode import SMB3EpisodeConfig, SMB3EpisodeManager
from neuroforge.game.loop import VisionOnlyGameLoop
from neuroforge.game.rewards import VisionMetricRewardConfig, VisionMetricRewardModel
from neuroforge.game.rewards_smb3 import SMB3RewardConfig, SMB3RewardModel

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
    "SMB3RewardConfig",
    "SMB3RewardModel",
    "ScriptedGameClient",
    "VisionMetricRewardConfig",
    "VisionMetricRewardModel",
    "VisionOnlyGameLoop",
]
