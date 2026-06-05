"""GameTrainingTask â€” play a game and learn from vision-only reward online.

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
from typing import TYPE_CHECKING, Any, cast

from neuroforge.agents.actuators.action_commitment import (
    TemporalCommitment,
    TemporalCommitmentConfig,
)
from neuroforge.agents.actuators.action_decoder import (
    ActionDecodeConfig,
    ActionDecoder,
)
from neuroforge.agents.brains.policy_network import (
    PolicyNetwork,
    PolicyNetworkConfig,
    build_policy_network,
)
from neuroforge.agents.brains.stateful_engine import CoreEnginePolicyEngine
from neuroforge.agents.policies.snn_policy import SNNGamePolicy
from neuroforge.applications.tasks.base import BaseTask
from neuroforge.construction.network_factory import to_topology_json
from neuroforge.contracts.applications.games import NINTENDO_BUTTONS
from neuroforge.environments.games.smb3.actions import ActionEnergyConfig, ActionEnergyModel
from neuroforge.environments.games.smb3.environment import VisionOnlyGameLoop
from neuroforge.environments.games.smb3.rewards import (
    VisionMetricRewardConfig,
    VisionMetricRewardModel,
)
from neuroforge.environments.games.smb3.state import PolicyCheckpoint
from neuroforge.learning.training_loop import OnlineRSTDPConfig, OnlineRSTDPTrainer
from neuroforge.perception.vision.encoding.frame_preprocess import (
    FramePreprocessConfig,
    FramePreprocessor,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.contracts.applications.games import (
        GameTransition,
        IEpisodeManager,
        IFrameMetricExtractor,
        IGameClient,
        IRewardModel,
    )
    from neuroforge.contracts.messaging import IEventBus
    from neuroforge.environments.games.smb3.curriculum import ICurriculum
    from neuroforge.perception.vision.encoding.frame_encoder import IFrameEncoder

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
    n_hidden_layers: int = 1
    motor_per_button: int = 4
    input_fanin: int = 64
    hidden_fanin: int = 0
    recurrent_hidden: bool = False
    input_to_motor_skip: bool = False
    init_scale: float = 0.5
    tau_mem: float = 5e-3
    # inference / exploration
    decide_ticks: int = 12
    noise_amp: float = 0.05
    commit_frames: int = 0             # >0 holds decoded intent (anti-dither momentum)
    commit_all_buttons: bool = False   # False = hold horizontal heading only
    action_energy: ActionEnergyConfig = field(default_factory=ActionEnergyConfig)
    dt: float = 1e-3
    # run length
    max_episodes: int = 100
    frames_per_episode: int = 2048
    # resume: load learned weights before training/evaluation (if the file exists).
    # By default this reads checkpoint_path; resume_checkpoint_path lets eval-only
    # runs load a learned policy without also using that path as the save target.
    resume: bool = False
    resume_checkpoint_path: str | None = None
    resume_allow_partial: bool = False
    # close_client=False borrows a shared client (e.g. one emulator reused across
    # many evolution genomes) instead of closing it at the end of the run.
    close_client: bool = True
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
        network_builder: Callable[[int], PolicyNetwork] | None = None,
    ) -> None:
        super().__init__(event_bus, stop_check)
        self._cfg = config or GameTrainingConfig()
        self._client = client
        self._extractor = metric_extractor
        self._reward_model = reward_model
        self._episode_manager = episode_manager
        self._curriculum = curriculum
        self._encoder = encoder
        # Optional injected phenotype builder (n_input -> PolicyNetwork). When set
        # (e.g. an evolved GraphGenome's graph), it replaces the fixed
        # config-driven build_policy_network â€” that is how invented topologies run.
        self._network_builder = network_builder
        self._active_encoder: object | None = None

    def run(self) -> GameTrainingResult:
        """Run the training loop and return a summary."""
        if self._client is None:
            msg = "GameTrainingTask.run() requires a client (BizHawk or scripted)"
            raise ValueError(msg)
        cfg = self._cfg

        preprocessor = self._encoder or FramePreprocessor(cfg.preprocess)
        self._active_encoder = preprocessor  # checkpointed alongside the policy
        if self._network_builder is not None:
            net = self._network_builder(preprocessor.input_size)
        else:
            net = build_policy_network(
                PolicyNetworkConfig(
                    n_input=preprocessor.input_size,
                    n_hidden=cfg.n_hidden,
                    n_hidden_layers=cfg.n_hidden_layers,
                    motor_per_button=cfg.motor_per_button,
                    input_fanin=cfg.input_fanin,
                    hidden_fanin=cfg.hidden_fanin,
                    recurrent_hidden=cfg.recurrent_hidden,
                    input_to_motor_skip=cfg.input_to_motor_skip,
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
            TemporalCommitment(
                TemporalCommitmentConfig(
                    commit_frames=cfg.commit_frames,
                    commit_all_buttons=cfg.commit_all_buttons,
                )
            )
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
        resume_info = self._maybe_resume(trainer)
        resumed = bool(resume_info.get("loaded", False))
        if resumed and cfg.rstdp.consolidation_strength > 0.0:
            trainer.anchor_current_weights()
        reward_model = self._reward_model or VisionMetricRewardModel(cfg.reward)
        action_energy = ActionEnergyModel(cfg.action_energy) if cfg.action_energy.enabled else None
        loop = VisionOnlyGameLoop(
            client=self._client,
            policy=policy,
            metric_extractor=self._extractor,
            reward_model=reward_model,
            episode_manager=self._episode_manager,
            close_client=cfg.close_client,
        )

        self._emit(
            "run_start", 0, "game_training",
            {
                "plastic": list(net.plastic_projections),
                "resumed": resumed,
                "resume": resume_info,
                "consolidation_strength": float(cfg.rstdp.consolidation_strength),
            },
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
            ep_energy_reward = 0.0
            ep_button_total = 0
            ep_change_total = 0
            ep_energy_min = cfg.action_energy.capacity
            button_counts = {name: 0 for name in NINTENDO_BUTTONS}
            if action_energy is not None:
                action_energy.begin_episode()
            for _ in range(cfg.frames_per_episode):
                if self._should_stop():
                    stopped = True
                    break
                transition = loop.step(current)
                energy_metrics: dict[str, float | int] = {}
                reward = transition.reward
                if action_energy is not None:
                    energy = action_energy.score(transition.action, current, transition.after)
                    reward += energy.reward_delta
                    ep_energy_reward += energy.reward_delta
                    ep_button_total += energy.button_count
                    ep_change_total += energy.change_count
                    ep_energy_min = min(ep_energy_min, energy.energy)
                    energy_metrics = {
                        "reward_action_energy": energy.reward_delta,
                        "action_energy": energy.energy,
                        "action_energy_cost": energy.cost,
                        "action_energy_shortage": energy.shortage,
                        "action_energy_refill": energy.refill,
                        "action_button_count": energy.button_count,
                        "action_change_count": energy.change_count,
                    }
                else:
                    pressed = set(transition.action.pressed())
                    ep_button_total += len(pressed)
                for name in transition.action.pressed():
                    button_counts[name] += 1
                telemetry = trainer.learn(reward)
                frame_total += 1
                ep_frames += 1
                ep_reward += reward
                ep_max_x = max(ep_max_x, transition.after.metrics.x_progress or 0.0)
                self._maybe_emit_scalar(
                    frame_total,
                    episode,
                    transition,
                    telemetry,
                    energy_metrics,
                )
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
                    "reward_action_energy_sum": ep_energy_reward,
                    "reward_baseline": trainer.reward_baseline,
                    "max_x_progress": ep_max_x,
                    "curriculum_stage": int(getattr(self._curriculum, "stage", 0)),
                    "action_button_mean": ep_button_total / ep_frames if ep_frames else 0.0,
                    "action_change_mean": ep_change_total / ep_frames if ep_frames else 0.0,
                    "action_energy_min": ep_energy_min if action_energy is not None else 0.0,
                    **{
                        f"action_{name.lower()}_frac": count / ep_frames if ep_frames else 0.0
                        for name, count in button_counts.items()
                    },
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

    #

    def _maybe_resume(self, trainer: OnlineRSTDPTrainer) -> dict[str, Any]:
        """Load learned weights from the checkpoint before training, if asked.

        Returns JSON-safe metadata so warm-start coverage is visible in run logs.
        """
        cfg = self._cfg
        info: dict[str, Any] = {
            "requested": bool(cfg.resume),
            "loaded": False,
            "path": "",
            "allow_partial": bool(cfg.resume_allow_partial),
            "reason": "",
        }
        if not cfg.resume:
            info["reason"] = "disabled"
            return info
        checkpoint_path = cfg.resume_checkpoint_path or cfg.checkpoint_path
        if checkpoint_path is None:
            info["reason"] = "no_path"
            return info
        info["path"] = str(checkpoint_path)
        if not Path(checkpoint_path).exists():
            info["reason"] = "missing"
            return info
        payload = PolicyCheckpoint.load(
            checkpoint_path,
            trainer=trainer,
            encoder=self._active_encoder,
            allow_partial=cfg.resume_allow_partial,
        )
        load_summary = payload.get("load_summary", {})
        info.update(
            {
                "loaded": True,
                "episode": int(payload.get("episode", 0)),
                "frame": int(payload.get("frame", 0)),
                "weights_copied": _sum_load_summary(load_summary, "weights", "copied_numel"),
                "weights_target": _sum_load_summary(load_summary, "weights", "target_numel"),
                "eligibility_copied": _sum_load_summary(
                    load_summary,
                    "eligibility",
                    "copied_numel",
                ),
                "eligibility_target": _sum_load_summary(
                    load_summary,
                    "eligibility",
                    "target_numel",
                ),
                "encoder_loaded": (
                    bool(cast("dict[str, Any]", load_summary).get("encoder_loaded", False))
                    if isinstance(load_summary, dict)
                    else False
                ),
                "load_summary": load_summary if isinstance(load_summary, dict) else {},
            }
        )
        return info

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
        self,
        frame: int,
        episode: int,
        transition: GameTransition,
        telemetry: dict[str, float],
        energy_metrics: dict[str, float | int],
    ) -> None:
        if self._cfg.telemetry_every <= 0 or frame % self._cfg.telemetry_every != 0:
            return
        metrics = transition.after.metrics
        action = transition.action
        data: dict[str, float | int] = {
            "frame": frame,
            "episode": episode,
            **telemetry,
            **energy_metrics,
            **{
                f"action_{name.lower()}": int(name in action.pressed())
                for name in NINTENDO_BUTTONS
            },
        }
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


def _sum_load_summary(load_summary: object, section: str, field: str) -> int:
    """Sum integer fields from one checkpoint load-summary section."""
    if not isinstance(load_summary, dict):
        return 0
    raw_section = cast("dict[str, Any]", load_summary).get(section, {})
    if not isinstance(raw_section, dict):
        return 0
    total = 0
    for raw_item in cast("dict[str, Any]", raw_section).values():
        if not isinstance(raw_item, dict):
            continue
        value = cast("dict[str, Any]", raw_item).get(field, 0)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            total += value
    return total
