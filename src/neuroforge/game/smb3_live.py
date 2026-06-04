# pyright: basic
"""Shared SMB3 live-training builders.

The live training, live evolution, and checkpoint-evaluation scripts should not
quietly drift into different reward or perception defaults. This module keeps
the SMB3-specific policy/runtime wiring in one small place while callers still
own machine-local paths and budgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from neuroforge.game.action_energy import ActionEnergyConfig
from neuroforge.game.curriculum import SMB3Curriculum
from neuroforge.game.episode import SMB3EpisodeConfig, SMB3EpisodeManager
from neuroforge.game.policies.action_decode import ActionDecodeConfig
from neuroforge.game.policies.preprocess import FramePreprocessConfig
from neuroforge.game.rewards import VisionMetricRewardConfig
from neuroforge.game.rewards_smb3 import SMB3RewardConfig, SMB3RewardModel
from neuroforge.game.vision import SMB3HudConfig, SMB3HudExtractor
from neuroforge.learning.online_rstdp import OnlineRSTDPConfig
from neuroforge.vision.encoding import (
    PerceptionStack,
    PerceptionStackConfig,
    RetinaEncoderConfig,
)

if TYPE_CHECKING:
    from neuroforge.tasks.game_training import GameTrainingConfig

__all__ = [
    "build_smb3_curriculum",
    "build_smb3_episode_manager",
    "build_smb3_game_training_config",
    "build_smb3_hud_extractor",
    "build_smb3_perception_stack",
    "build_smb3_reward_model",
    "existing_savestates",
]


def existing_savestates(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Return configured savestates that currently exist on disk."""
    return tuple(path for path in paths if Path(path).exists())


def build_smb3_game_training_config(
    *,
    max_episodes: int,
    frames_per_episode: int,
    telemetry_every: int = 0,
    checkpoint_every: int = 0,
    checkpoint_path: str | None = None,
    resume: bool = False,
    resume_checkpoint_path: str | None = None,
    consolidation_strength: float = 0.0,
    plastic: bool = True,
    deterministic: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
) -> GameTrainingConfig:
    """Return the canonical SMB3 game-training config.

    ``plastic=False`` freezes R-STDP weight updates. ``deterministic=True`` also
    disables motor noise and uses threshold decoding, which makes checkpoint
    comparisons repeatable. Training should normally use the stochastic defaults.
    """
    from neuroforge.tasks.game_training import GameTrainingConfig

    base_rstdp = OnlineRSTDPConfig()
    return GameTrainingConfig(
        preprocess=FramePreprocessConfig(
            out_h=28,
            out_w=32,
            motion=True,
            device=device,
            dtype=dtype,
        ),
        decode=(
            ActionDecodeConfig(
                mode="threshold",
                threshold=0.45,
                dpad_explore_floor=0.0,
                max_pressed=3,
            )
            if deterministic
            else ActionDecodeConfig(
                mode="bernoulli",
                threshold=0.45,
                temperature=0.2,
                dpad_explore_floor=0.05,
                max_pressed=3,
            )
        ),
        rstdp=OnlineRSTDPConfig(
            lr=base_rstdp.lr if plastic else 0.0,
            reward_scale=0.05,
            consolidation_strength=max(0.0, float(consolidation_strength)),
        ),
        noise_amp=0.0 if deterministic else 0.25,
        commit_frames=4,
        commit_all_buttons=True,
        action_energy=ActionEnergyConfig(
            enabled=True,
            capacity=8.0,
            recover_per_frame=0.035,
            button_cost=0.025,
            change_cost=0.10,
            cost_penalty_scale=0.30,
            shortage_penalty_scale=1.50,
            progress_refill_scale=6.0,
        ),
        max_episodes=max(1, int(max_episodes)),
        frames_per_episode=max(1, int(frames_per_episode)),
        telemetry_every=max(0, int(telemetry_every)),
        checkpoint_every=max(0, int(checkpoint_every)),
        checkpoint_path=checkpoint_path,
        resume=resume,
        resume_checkpoint_path=resume_checkpoint_path,
        device=device,
        dtype=dtype,
    )


def build_smb3_perception_stack(
    *,
    learn: bool = True,
    device: str = "cpu",
    dtype: str = "float32",
) -> PerceptionStack:
    """Build the SMB3 bio-perception stack used by the policy."""
    return PerceptionStack(
        PerceptionStackConfig(
            retina=RetinaEncoderConfig(out_h=28, out_w=32, device=device, dtype=dtype),
            features=True,
            objects=True,
            motion=True,
            learn=learn,
        ),
    )


def build_smb3_hud_extractor() -> SMB3HudExtractor:
    """Build the calibrated HUD/progress extractor for SMB3 frames."""
    return SMB3HudExtractor(SMB3HudConfig(track_progress=True))


def build_smb3_episode_manager() -> SMB3EpisodeManager:
    """Build the vision-derived SMB3 episode terminator."""
    return SMB3EpisodeManager(SMB3EpisodeConfig())


def build_smb3_reward_model() -> SMB3RewardModel:
    """Build the canonical SMB3 vision reward model."""
    return SMB3RewardModel(
        SMB3RewardConfig(
            base=VisionMetricRewardConfig(
                progress_scale=200.0,
                score_scale=0.05,
                time_delta_scale=0.0,
                life_loss_penalty=-50.0,
                life_gain_bonus=25.0,
            ),
            idle_penalty=-0.05,
            stall_penalty=-3.0,
        ),
    )


def build_smb3_curriculum(
    savestate_paths: tuple[str, ...],
    *,
    advance_threshold: float = 0.9,
    min_episodes_per_stage: int = 4,
) -> SMB3Curriculum:
    """Build the savestate curriculum shared by live SMB3 entrypoints."""
    return SMB3Curriculum(
        savestate_paths,
        advance_threshold=advance_threshold,
        min_episodes_per_stage=min_episodes_per_stage,
    )
