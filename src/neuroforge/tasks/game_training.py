# pyright: basic
"""GameTrainingTask — play a game and learn from vision-only reward online.

Closes the loop: a stateful spiking policy (Phase 2) drives the game through the
:class:`VisionOnlyGameLoop`, the vision-metric reward (Phase 1) scores each
transition, and :class:`OnlineRSTDPTrainer` modulates the synapses with that
reward during play. Eligibility traces advance every engine tick (via the
policy engine's ``on_tick`` hook); the reward is applied once per frame.

It emits the standard monitor events (``run_start``/``topology``/``scalar``/
``training_trial``/``training_end``/``run_end``) so the existing dashboard,
artifact writer, and monitors work unchanged. A concrete client is injected so
the same task drives BizHawk live or a scripted client offline/CI.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from neuroforge.game.checkpoint import PolicyCheckpoint
from neuroforge.game.loop import VisionOnlyGameLoop
from neuroforge.game.policies.action_decode import ActionDecodeConfig, ActionDecoder
from neuroforge.game.policies.commitment import TemporalCommitment, TemporalCommitmentConfig
from neuroforge.game.policies.network import PolicyNetworkConfig, build_policy_network
from neuroforge.game.policies.preprocess import FramePreprocessConfig, FramePreprocessor
from neuroforge.game.policies.snn_policy import SNNGamePolicy
from neuroforge.game.policies.stateful_engine import CoreEnginePolicyEngine
from neuroforge.game.rewards import VisionMetricRewardConfig, VisionMetricRewardModel
from neuroforge.learning.online_rstdp import OnlineRSTDPConfig, OnlineRSTDPTrainer
from neuroforge.network.factory import to_topology_json
from neuroforge.tasks.base import BaseTask

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.game import (
        IEpisodeManager,
        IFrameMetricExtractor,
        IGameClient,
        IRewardModel,
    )
    from neuroforge.contracts.monitors import IEventBus
    from neuroforge.game.curriculum import ICurriculum
    from neuroforge.game.policies.encoder import IFrameEncoder

__all__ = ["GameTrainingConfig", "GameTrainingResult", "GameTrainingTask"]


@dataclass(frozen=True, slots=True)
class GameTrainingConfig:
    """Configuration for :class:`GameTrainingTask`."""

    preprocess: FramePreprocessConfig = field(default_factory=FramePreprocessConfig)
    decode: ActionDecodeConfig = field(default_factory=ActionDecodeConfig)
    rstdp: OnlineRSTDPConfig = field(default_factory=OnlineRSTDPConfig)
    reward: VisionMetricRewardConfig = field(default_factory=VisionMetricRewardConfig)
    # network shape
    n_hidden: int = 128
    motor_per_button: int = 4
    input_fanin: int = 64
    recurrent_hidden: bool = False
    init_scale: float = 0.5
    tau_mem: float = 5e-3
    # inference / exploration
    decide_ticks: int = 12
    noise_amp: float = 0.05
    commit_frames: int = 0             # >0 holds the decoded heading (anti-dither momentum)
    dt: float = 1e-3
    # run length
    max_episodes: int = 100
    frames_per_episode: int = 2048
    # resume: load learned weights from checkpoint_path before training (if it exists)
    resume: bool = False
    # bookkeeping
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    telemetry_every: int = 4
    checkpoint_every: int = 0          # 0 = no periodic checkpoints
    checkpoint_path: str | None = None


@dataclass
class GameTrainingResult:
    """Summary returned by :meth:`GameTrainingTask.run`."""

    episodes: int
    frames: int
    reward_baseline: float
    stopped: bool


class GameTrainingTask(BaseTask):
    """Train a spiking policy to play a game via online reward-modulated STDP."""

    def __init__(
        self,
        config: GameTrainingConfig | None = None,
        event_bus: IEventBus | None = None,
        *,
        stop_check: Callable[[], bool] | None = None,
        client: IGameClient | None = None,
        metric_extractor: IFrameMetricExtractor | None = None,
        reward_model: IRewardModel | None = None,
        episode_manager: IEpisodeManager | None = None,
        curriculum: ICurriculum | None = None,
        encoder: IFrameEncoder | None = None,
    ) -> None:
        super().__init__(event_bus, stop_check)
        self._cfg = config or GameTrainingConfig()
        self._client = client
        self._extractor = metric_extractor
        self._reward_model = reward_model
        self._episode_manager = episode_manager
        self._curriculum = curriculum
        self._encoder = encoder
        self._active_encoder: object | None = None

    def run(self) -> GameTrainingResult:
        """Run the training loop and return a summary."""
        if self._client is None:
            msg = "GameTrainingTask.run() requires a client (BizHawk or scripted)"
            raise ValueError(msg)
        cfg = self._cfg

        preprocessor = self._encoder or FramePreprocessor(cfg.preprocess)
        self._active_encoder = preprocessor  # checkpointed alongside the policy
        net = build_policy_network(
            PolicyNetworkConfig(
                n_input=preprocessor.input_size,
                n_hidden=cfg.n_hidden,
                motor_per_button=cfg.motor_per_button,
                input_fanin=cfg.input_fanin,
                recurrent_hidden=cfg.recurrent_hidden,
                init_scale=cfg.init_scale,
                tau_mem=cfg.tau_mem,
                dt=cfg.dt,
                seed=cfg.seed,
                device=cfg.device,
                dtype=cfg.dtype,
            ),
        )
        policy_engine = CoreEnginePolicyEngine(
            net.engine,
            motor_pop=net.motor_pop,
            motor_per_button=net.motor_per_button,
            n_buttons=net.n_buttons,
            input_pop=net.input_pop,
            noise_amp=cfg.noise_amp,
            seed=cfg.seed,
        )
        trainer = OnlineRSTDPTrainer(
            net.engine,
            dataclasses.replace(
                cfg.rstdp, dt=cfg.dt, plastic_projections=net.plastic_projections,
            ),
            device=cfg.device,
            dtype=cfg.dtype,
        )
        policy_engine.set_on_tick(trainer.accumulate)
        commitment = (
            TemporalCommitment(TemporalCommitmentConfig(commit_frames=cfg.commit_frames))
            if cfg.commit_frames > 0
            else None
        )
        policy = SNNGamePolicy(
            engine=policy_engine,
            preprocessor=preprocessor,
            decoder=ActionDecoder(cfg.decode, seed=cfg.seed),
            decide_ticks=cfg.decide_ticks,
            commitment=commitment,
        )
        resumed = self._maybe_resume(trainer)
        reward_model = self._reward_model or VisionMetricRewardModel(cfg.reward)
        loop = VisionOnlyGameLoop(
            client=self._client,
            policy=policy,
            metric_extractor=self._extractor,
            reward_model=reward_model,
            episode_manager=self._episode_manager,
        )

        self._emit(
            "run_start", 0, "game_training",
            {"plastic": list(net.plastic_projections), "resumed": resumed},
        )
        self._emit("topology", 0, "game_training", to_topology_json(net.engine))

        frame_total = 0
        episodes_done = 0
        stopped = False
        for episode in range(cfg.max_episodes):
            if self._should_stop():
                stopped = True
                break
            self._begin_episode_savestate(episode)
            policy.begin_episode()
            trainer.begin_episode()
            current = loop.reset()
            ep_frames = 0
            ep_reward = 0.0
            ep_max_x = 0.0
            for _ in range(cfg.frames_per_episode):
                if self._should_stop():
                    stopped = True
                    break
                transition = loop.step(current)
                telemetry = trainer.learn(transition.reward)
                frame_total += 1
                ep_frames += 1
                ep_reward += transition.reward
                ep_max_x = max(ep_max_x, transition.after.metrics.x_progress or 0.0)
                self._maybe_emit_scalar(frame_total, episode, transition, telemetry)
                self._maybe_checkpoint(trainer, episode, frame_total)
                current = transition.after
                if transition.done:
                    break
            if self._curriculum is not None:
                self._curriculum.report_episode(ep_max_x)
            episodes_done += 1
            self._emit(
                "training_trial", episode, "game_training",
                {
                    "episode": episode,
                    "frames": ep_frames,
                    "reward_sum": ep_reward,
                    "reward_mean": ep_reward / ep_frames if ep_frames else 0.0,
                    "reward_baseline": trainer.reward_baseline,
                    "max_x_progress": ep_max_x,
                    "curriculum_stage": int(getattr(self._curriculum, "stage", 0)),
                },
            )
            if stopped:
                break

        loop.close()
        if cfg.checkpoint_path is not None:
            PolicyCheckpoint.save(
                cfg.checkpoint_path, trainer=trainer, episode=episodes_done,
                frame=frame_total, encoder=self._active_encoder,
            )
        end_data = {
            "episodes": episodes_done,
            "frames": frame_total,
            "reward_baseline": trainer.reward_baseline,
            "stopped": stopped,
        }
        self._emit("training_end", frame_total, "game_training", end_data)
        self._emit("run_end", frame_total, "game_training", end_data)
        return GameTrainingResult(
            episodes=episodes_done,
            frames=frame_total,
            reward_baseline=trainer.reward_baseline,
            stopped=stopped,
        )

    # ── helpers ───────────────────────────────────────────────────────

    def _maybe_resume(self, trainer: OnlineRSTDPTrainer) -> bool:
        """Load learned weights from the checkpoint before training, if asked.

        Returns whether a checkpoint was loaded — so a run can *continue* learning
        across invocations instead of always restarting from random weights.
        """
        cfg = self._cfg
        if not cfg.resume or cfg.checkpoint_path is None:
            return False
        if not Path(cfg.checkpoint_path).exists():
            return False
        PolicyCheckpoint.load(cfg.checkpoint_path, trainer=trainer, encoder=self._active_encoder)
        return True

    def _begin_episode_savestate(self, episode: int) -> None:
        """Ask the curriculum where this episode starts and queue it on the client.

        The client is duck-typed: only BizHawk supports savestate resets, so a
        scripted/CI client (no ``queue_savestate``) simply ignores the request.
        """
        if self._curriculum is None:
            return
        savestate = self._curriculum.savestate_for(episode)
        if savestate is None:
            return
        queue = getattr(self._client, "queue_savestate", None)
        if callable(queue):
            queue(savestate)

    def _maybe_emit_scalar(
        self, frame: int, episode: int, transition: object, telemetry: dict[str, float],
    ) -> None:
        if self._cfg.telemetry_every <= 0 or frame % self._cfg.telemetry_every != 0:
            return
        metrics = transition.after.metrics  # type: ignore[attr-defined]
        data: dict[str, float | int] = {"frame": frame, "episode": episode, **telemetry}
        if metrics.x_progress is not None:
            data["x_progress"] = float(metrics.x_progress)
        if metrics.score is not None:
            data["score"] = int(metrics.score)
        if metrics.lives is not None:
            data["lives"] = int(metrics.lives)
        self._emit("scalar", frame, "game_training", data)

    def _maybe_checkpoint(
        self, trainer: OnlineRSTDPTrainer, episode: int, frame: int,
    ) -> None:
        cfg = self._cfg
        if cfg.checkpoint_path is None or cfg.checkpoint_every <= 0:
            return
        if frame % cfg.checkpoint_every == 0:
            PolicyCheckpoint.save(
                cfg.checkpoint_path, trainer=trainer, episode=episode,
                frame=frame, encoder=self._active_encoder,
            )
