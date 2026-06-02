# pyright: basic
"""Game-backed fitness evaluators for policy evolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neuroforge.evolution.evaluators import GameTrainingFitnessEvaluator
from neuroforge.game.clients.launcher import EmuHawkLauncher
from neuroforge.game.episode import SMB3EpisodeConfig, SMB3EpisodeManager
from neuroforge.game.policies.action_decode import ActionDecodeConfig
from neuroforge.game.policies.preprocess import FramePreprocessConfig
from neuroforge.game.rewards import VisionMetricRewardConfig, VisionMetricRewardModel
from neuroforge.game.rewards_smb3 import SMB3RewardConfig, SMB3RewardModel
from neuroforge.learning.online_rstdp import OnlineRSTDPConfig
from neuroforge.tasks.game_training import GameTrainingConfig
from neuroforge.vision.encoding import (
    PerceptionStack,
    PerceptionStackConfig,
    RetinaEncoderConfig,
)

__all__ = [
    "SMB3LiveFitnessConfig",
    "ScriptedGameFitnessConfig",
    "build_live_smb3_fitness_evaluator",
    "build_scripted_progress_fitness_evaluator",
    "existing_savestates",
]


@dataclass(frozen=True, slots=True)
class ScriptedGameFitnessConfig:
    """Configuration for fast offline game-backed evolution fitness."""

    max_episodes: int = 1
    frames_per_episode: int = 120
    telemetry_every: int = 0
    width: int = 32
    height: int = 28
    channels: int = 1
    device: str = "cpu"
    dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class SMB3LiveFitnessConfig:
    """Configuration for short live BizHawk SMB3 fitness evaluations."""

    emuhawk_path: str
    rom_path: str
    lua_script: str
    savestate_paths: tuple[str, ...] = ()
    port: int = 8650
    frameskip: int = 4
    speed_percent: int = 400
    max_episodes: int = 1
    frames_per_episode: int = 600
    telemetry_every: int = 0
    connect_timeout_s: float = 60.0
    step_timeout_s: float = 45.0
    launch: bool = True
    device: str = "cpu"
    dtype: str = "float32"


def existing_savestates(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Return configured savestates that currently exist on disk."""
    return tuple(path for path in paths if Path(path).exists())


def build_scripted_progress_fitness_evaluator(
    config: ScriptedGameFitnessConfig | None = None,
) -> GameTrainingFitnessEvaluator:
    """Build an offline action-progress evaluator through ``GameTrainingTask``."""
    from neuroforge.game.clients.scripted import ActionProgressGameClient

    cfg = config or ScriptedGameFitnessConfig()
    base = _base_game_config(
        max_episodes=cfg.max_episodes,
        frames_per_episode=cfg.frames_per_episode,
        telemetry_every=cfg.telemetry_every,
        device=cfg.device,
        dtype=cfg.dtype,
    )
    return GameTrainingFitnessEvaluator(
        client_factory=lambda: ActionProgressGameClient(
            width=cfg.width,
            height=cfg.height,
            channels=cfg.channels,
            max_steps=cfg.frames_per_episode,
        ),
        base_config=base,
        reward_model_factory=lambda: VisionMetricRewardModel(
            VisionMetricRewardConfig(
                progress_scale=200.0,
                score_scale=0.0,
                life_loss_penalty=0.0,
                missing_metric_reward=0.0,
            )
        ),
    )


def build_live_smb3_fitness_evaluator(
    config: SMB3LiveFitnessConfig,
) -> GameTrainingFitnessEvaluator:
    """Build a short live SMB3 evaluator backed by BizHawk/GameTrainingTask."""
    from neuroforge.game import BizHawkClient, BizHawkClientConfig
    from neuroforge.game.curriculum import SMB3Curriculum
    from neuroforge.game.vision import SMB3HudConfig, SMB3HudExtractor

    base = _base_game_config(
        max_episodes=config.max_episodes,
        frames_per_episode=config.frames_per_episode,
        telemetry_every=config.telemetry_every,
        device=config.device,
        dtype=config.dtype,
    )
    savestates = existing_savestates(config.savestate_paths)
    return GameTrainingFitnessEvaluator(
        client_factory=lambda: BizHawkClient(
            BizHawkClientConfig(
                port=config.port,
                width=256,
                height=240,
                channels=3,
                frameskip=config.frameskip,
                connect_timeout_s=config.connect_timeout_s,
                step_timeout_s=config.step_timeout_s,
                launch=config.launch,
            ),
            launcher=EmuHawkLauncher(
                emuhawk_path=config.emuhawk_path,
                lua_script=config.lua_script,
                rom_path=config.rom_path,
                frameskip=config.frameskip,
                speed_percent=config.speed_percent,
            ),
        ),
        base_config=base,
        metric_extractor_factory=lambda: SMB3HudExtractor(SMB3HudConfig(track_progress=True)),
        reward_model_factory=lambda: SMB3RewardModel(
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
            )
        ),
        episode_manager_factory=lambda: SMB3EpisodeManager(SMB3EpisodeConfig()),
        curriculum_factory=lambda: SMB3Curriculum(
            savestates,
            advance_threshold=0.9,
            min_episodes_per_stage=4,
        ),
        encoder_factory=_build_perception_stack,
    )


def _base_game_config(
    *,
    max_episodes: int,
    frames_per_episode: int,
    telemetry_every: int,
    device: str,
    dtype: str,
) -> GameTrainingConfig:
    return GameTrainingConfig(
        preprocess=FramePreprocessConfig(
            out_h=28,
            out_w=32,
            motion=True,
            device=device,
            dtype=dtype,
        ),
        decode=ActionDecodeConfig(
            mode="bernoulli",
            threshold=0.25,
            temperature=0.3,
            dpad_explore_floor=0.1,
        ),
        rstdp=OnlineRSTDPConfig(reward_scale=0.05),
        noise_amp=0.4,
        commit_frames=8,
        max_episodes=max(1, int(max_episodes)),
        frames_per_episode=max(1, int(frames_per_episode)),
        telemetry_every=max(0, int(telemetry_every)),
        checkpoint_every=0,
        checkpoint_path=None,
        resume=False,
        device=device,
        dtype=dtype,
    )


def _build_perception_stack() -> PerceptionStack:
    return PerceptionStack(
        PerceptionStackConfig(
            retina=RetinaEncoderConfig(out_h=28, out_w=32),
            features=True,
            objects=True,
            motion=True,
            learn=True,
        ),
    )
