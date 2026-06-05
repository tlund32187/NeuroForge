"""Game-backed fitness evaluators for policy evolution."""

from __future__ import annotations

import dataclasses
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neuroforge.neuroevolution.fitness.evaluators import GameTrainingFitnessEvaluator

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.applications.tasks.game_training import GameTrainingConfig
    from neuroforge.perception.vision.encoding.frame_encoder import IFrameEncoder
from neuroforge.environments.games.clients.bizhawk.launcher import EmuHawkLauncher
from neuroforge.environments.games.smb3.adapters.bizhawk_smb3_adapter import (
    build_smb3_curriculum,
    build_smb3_episode_manager,
    build_smb3_game_training_config,
    build_smb3_hud_extractor,
    build_smb3_perception_stack,
    build_smb3_reward_model,
    existing_savestates,
)
from neuroforge.environments.games.smb3.episode import SMB3EpisodeConfig
from neuroforge.environments.games.smb3.rewards import (
    VisionMetricRewardConfig,
    VisionMetricRewardModel,
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
    bridge_error_path: str | None = None
    eval_repeats: int = 2       # average N short rollouts/genome to cut fitness noise
    perf_telemetry: bool = False
    stall_patience: int = 600
    min_progress_frames: int = 0
    min_progress: float = 0.0
    max_decide_ticks: int = 0
    fitness_progress_scale: float = 100.0
    fitness_survival_scale: float = 1.0
    fitness_durable_progress_weight: float = 1.0
    device: str = "cpu"
    dtype: str = "float32"


def build_scripted_progress_fitness_evaluator(
    config: ScriptedGameFitnessConfig | None = None,
) -> GameTrainingFitnessEvaluator:
    """Build an offline action-progress evaluator through ``GameTrainingTask``."""
    from neuroforge.environments.games.clients.scripted import ActionProgressGameClient

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
    *,
    encoder_factory: Callable[[], IFrameEncoder] | None = None,
) -> GameTrainingFitnessEvaluator:
    """Build a short live SMB3 evaluator backed by BizHawk/GameTrainingTask.

    ``encoder_factory`` overrides the perception front-end; it defaults to the full
    A0+A1+A2+A3 :func:`build_smb3_perception_stack`. HyperNEAT runs pass a plain
    retinotopic :class:`RetinaEncoder` so the input substrate is a true 2-D grid.
    """
    from neuroforge.environments.games.smb3 import BizHawkClient, BizHawkClientConfig

    base = build_smb3_game_training_config(
        max_episodes=config.max_episodes,
        frames_per_episode=config.frames_per_episode,
        telemetry_every=config.telemetry_every,
        perf_telemetry=config.perf_telemetry,
        device=config.device,
        dtype=config.dtype,
    )
    savestates = existing_savestates(config.savestate_paths)
    episode_config = SMB3EpisodeConfig(
        stall_patience=max(1, int(config.stall_patience)),
        min_progress_frames=max(0, int(config.min_progress_frames)),
        min_progress=max(0.0, float(config.min_progress)),
    )
    bridge_error_path = config.bridge_error_path or str(
        Path(tempfile.gettempdir()) / f"neuroforge_bridge_error_{config.port}.log"
    )

    def _transform_config(
        base_config: GameTrainingConfig,
        _genome: object,
    ) -> GameTrainingConfig:
        if config.max_decide_ticks <= 0:
            return base_config
        decide_ticks = int(getattr(base_config, "decide_ticks", 0))
        if decide_ticks <= config.max_decide_ticks:
            return base_config
        return dataclasses.replace(
            base_config,
            decide_ticks=int(config.max_decide_ticks),
        )

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
                bridge_error_path=bridge_error_path,
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
        episode_manager_factory=lambda: build_smb3_episode_manager(episode_config),
        curriculum_factory=lambda: build_smb3_curriculum(
            savestates,
            advance_threshold=0.9,
            min_episodes_per_stage=4,
        ),
        encoder_factory=encoder_factory or build_smb3_perception_stack,
        # One launched emulator, reset per genome — not one cold-start per genome.
        reuse_client=True,
        eval_repeats=config.eval_repeats,
        config_transform=_transform_config,
        progress_scale=config.fitness_progress_scale,
        survival_scale=config.fitness_survival_scale,
        durable_progress_weight=config.fitness_durable_progress_weight,
    )
