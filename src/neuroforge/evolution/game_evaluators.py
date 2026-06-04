# pyright: basic
"""Game-backed fitness evaluators for policy evolution."""

from __future__ import annotations

from dataclasses import dataclass

from neuroforge.evolution.evaluators import GameTrainingFitnessEvaluator
from neuroforge.game.clients.launcher import EmuHawkLauncher
from neuroforge.game.rewards import VisionMetricRewardConfig, VisionMetricRewardModel
from neuroforge.game.smb3_live import (
    build_smb3_curriculum,
    build_smb3_episode_manager,
    build_smb3_game_training_config,
    build_smb3_hud_extractor,
    build_smb3_perception_stack,
    build_smb3_reward_model,
    existing_savestates,
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
    eval_repeats: int = 2       # average N short rollouts/genome to cut fitness noise
    device: str = "cpu"
    dtype: str = "float32"


def build_scripted_progress_fitness_evaluator(
    config: ScriptedGameFitnessConfig | None = None,
) -> GameTrainingFitnessEvaluator:
    """Build an offline action-progress evaluator through ``GameTrainingTask``."""
    from neuroforge.game.clients.scripted import ActionProgressGameClient

    cfg = config or ScriptedGameFitnessConfig()
    base = build_smb3_game_training_config(
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

    base = build_smb3_game_training_config(
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
                transport="socket",
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
        metric_extractor_factory=build_smb3_hud_extractor,
        reward_model_factory=build_smb3_reward_model,
        episode_manager_factory=build_smb3_episode_manager,
        curriculum_factory=lambda: build_smb3_curriculum(
            savestates,
            advance_threshold=0.9,
            min_episodes_per_stage=4,
        ),
        encoder_factory=build_smb3_perception_stack,
        # One launched emulator, reset per genome — not one cold-start per genome.
        reuse_client=True,
        eval_repeats=config.eval_repeats,
    )
