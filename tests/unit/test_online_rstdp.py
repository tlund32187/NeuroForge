"""Unit tests for the online R-STDP learning loop (Phase 3).

Drives the trainer with controlled pre/post spikes to verify the learning
mechanics directly: reward sign moves weights the right way, Dale-safe clamping
keeps magnitudes non-negative, eligibility bridges an action->reward delay, the
reward shaper computes a clipped RPE, checkpoints round-trip, and the full task
runs offline and honours stop requests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from neuroforge.agents.brains.policy_network import (
    PolicyNetworkConfig,
    build_policy_network,
)
from neuroforge.applications.tasks.game_training import GameTrainingConfig, GameTrainingTask
from neuroforge.biology.plasticity.rules.rstdp import RSTDPRule
from neuroforge.contracts.applications.games import (
    ControllerAction,
    EpisodeDecision,
    GameObservation,
    ScreenFrame,
    VisionGameMetrics,
)
from neuroforge.contracts.messaging import EventTopic
from neuroforge.environments.games.clients.scripted import ScriptedGameClient
from neuroforge.environments.games.smb3.actions import ActionEnergyConfig, ActionEnergyModel
from neuroforge.environments.games.smb3.state import PolicyCheckpoint, resume_status_lines
from neuroforge.learning.training_loop import (
    OnlineRSTDPConfig,
    OnlineRSTDPTrainer,
    RewardShaper,
)
from neuroforge.messaging.bus import EventBus
from neuroforge.perception.vision.encoding.frame_preprocess import FramePreprocessConfig

if TYPE_CHECKING:
    from pathlib import Path


def _small_net() -> object:
    return build_policy_network(
        PolicyNetworkConfig(n_input=8, n_hidden=8, motor_per_button=1, input_fanin=0, seed=1),
    )


def _trainer(net: object, **overrides: object) -> OnlineRSTDPTrainer:
    cfg = OnlineRSTDPConfig(
        lr=0.1, reward_scale=1.0, reward_clip=1.0, baseline_beta=0.0,
        plastic_projections=("in_to_hidden",), **overrides,  # type: ignore[arg-type]
    )
    return OnlineRSTDPTrainer(net.engine, cfg, device="cpu", dtype="float32")  # type: ignore[attr-defined]


def _force_spikes(net: object, *, pre: bool, post: bool) -> None:
    engine = net.engine  # type: ignore[attr-defined]
    engine.populations["input"].state["last_spikes"] = torch.full((8,), pre, dtype=torch.bool)
    engine.populations["hidden"].state["last_spikes"] = torch.full((8,), post, dtype=torch.bool)


def _metric_observation(step: int, *, x: float = 0.0, score: int = 0) -> GameObservation:
    frame = ScreenFrame(width=1, height=1, channels=1, data=b"\x00", frame_id=step)
    return GameObservation(
        step=step,
        t=step / 60.0,
        frame=frame,
        metrics=VisionGameMetrics(score=score, lives=3, x_progress=x),
    )


#


@pytest.mark.unit
def test_action_energy_penalizes_churn_and_refills_on_progress() -> None:
    model = ActionEnergyModel(
        ActionEnergyConfig(
            enabled=True,
            capacity=0.10,
            recover_per_frame=0.0,
            button_cost=0.08,
            change_cost=0.05,
            cost_penalty_scale=1.0,
            shortage_penalty_scale=2.0,
            progress_refill_scale=1.0,
        )
    )
    model.begin_episode()

    first = model.score(
        ControllerAction(right=True),
        _metric_observation(0, x=0.0),
        _metric_observation(1, x=0.0),
    )
    second = model.score(
        ControllerAction(up=True, right=True),
        _metric_observation(1, x=0.0),
        _metric_observation(2, x=0.2),
    )

    assert first.button_count == 1
    assert second.button_count == 2
    assert second.change_count == 1
    assert first.reward_delta < 0.0
    assert second.refill > 0.0
    assert second.shortage >= 0.0


@pytest.mark.unit
def test_rstdp_state_tensors_exposes_eligibility() -> None:
    rule = RSTDPRule()
    state = rule.init_state(4, "cpu", "float32")
    assert "eligibility" in rule.state_tensors(state)


@pytest.mark.unit
def test_reward_increases_then_punishment_clamps_to_zero() -> None:
    net = _small_net()
    trainer = _trainer(net)
    topo = net.engine.projections["in_to_hidden"].topology  # type: ignore[attr-defined]

    # Pre fires, post silent -> positive eligibility tags.
    _force_spikes(net, pre=True, post=False)
    for _ in range(3):
        trainer.accumulate()

    w0 = topo.weights.clone()
    trainer.learn(1.0)  # positive reward -> potentiate
    assert bool((topo.weights >= w0 - 1e-6).all())
    assert bool((topo.weights > w0 + 1e-6).any())

    # Hammer with negative reward -> weights driven down but never below 0 (Dale-safe).
    for _ in range(100):
        trainer.accumulate()
        trainer.learn(-1.0)
    assert bool((topo.weights >= 0.0).all())
    assert float(topo.weights.min()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_eligibility_bridges_action_reward_delay() -> None:
    net = _small_net()
    trainer = _trainer(net, tau_e=80e-3, dt=1e-3)

    _force_spikes(net, pre=True, post=False)
    for _ in range(4):
        trainer.accumulate()
    built = float(trainer.eligibility_snapshot()["in_to_hidden"].abs().sum())

    # No firing for several ticks: trace decays but must survive the gap.
    _force_spikes(net, pre=False, post=False)
    for _ in range(6):
        trainer.accumulate()
    decayed = float(trainer.eligibility_snapshot()["in_to_hidden"].abs().sum())

    assert 0.0 < decayed < built  # decayed but not gone -> bridges the delay


@pytest.mark.unit
def test_reward_shaper_rpe_and_clip() -> None:
    shaper = RewardShaper(scale=0.5, clip=1.0, beta=0.0)
    assert shaper.shape(4.0) == pytest.approx(1.0)   # 4*0.5=2 clipped to 1
    assert shaper.shape(-10.0) == pytest.approx(-1.0)
    # With a baseline, steady reward trends toward zero advantage.
    s = RewardShaper(scale=1.0, clip=10.0, beta=0.5)
    first = s.shape(1.0)
    second = s.shape(1.0)
    assert first > second  # baseline rises, advantage shrinks


#


@pytest.mark.unit
def test_checkpoint_round_trip(tmp_path: object) -> None:
    net = _small_net()
    trainer = _trainer(net)
    _force_spikes(net, pre=True, post=False)
    for _ in range(3):
        trainer.accumulate()
    trainer.learn(1.0)  # change weights
    saved = trainer.weights_snapshot()["in_to_hidden"].clone()

    path = tmp_path / "ckpt.pt"  # type: ignore[operator]
    PolicyCheckpoint.save(path, trainer=trainer, episode=2, frame=5)

    # Fresh network with different random weights; load should overwrite them.
    net2 = build_policy_network(
        PolicyNetworkConfig(n_input=8, n_hidden=8, motor_per_button=1, input_fanin=0, seed=999),
    )
    trainer2 = _trainer(net2)
    meta = PolicyCheckpoint.load(path, trainer=trainer2)
    assert meta["episode"] == 2
    loaded = trainer2.weights_snapshot()["in_to_hidden"]
    assert torch.allclose(loaded, saved)


@pytest.mark.unit
def test_checkpoint_partial_load_warm_starts_changed_shape(tmp_path: object) -> None:
    source_net = _small_net()
    source_trainer = _trainer(source_net)
    source_weights = torch.linspace(
        0.0,
        1.0,
        source_trainer.weights_snapshot()["in_to_hidden"].numel(),
        dtype=torch.float32,
    )
    source_topo = source_net.engine.projections["in_to_hidden"].topology  # type: ignore[attr-defined]
    source_topo.weights.copy_(source_weights)

    path = tmp_path / "partial.pt"  # type: ignore[operator]
    PolicyCheckpoint.save(path, trainer=source_trainer)

    target_net = build_policy_network(
        PolicyNetworkConfig(
            n_input=8,
            n_hidden=12,
            motor_per_button=1,
            input_fanin=0,
            seed=999,
        ),
    )
    target_trainer = OnlineRSTDPTrainer(
        target_net.engine,
        OnlineRSTDPConfig(lr=0.1, plastic_projections=("in_to_hidden",)),
        device="cpu",
        dtype="float32",
    )

    meta = PolicyCheckpoint.load(path, trainer=target_trainer, allow_partial=True)
    loaded = target_trainer.weights_snapshot()["in_to_hidden"].reshape(-1)
    n = source_weights.numel()
    weight_summary = meta["load_summary"]["weights"]["in_to_hidden"]
    eligibility_summary = meta["load_summary"]["eligibility"]["in_to_hidden"]
    assert loaded.numel() > n
    assert torch.allclose(loaded[:n], source_weights)
    assert weight_summary["partial"] is True
    assert weight_summary["source_numel"] == n
    assert weight_summary["copied_numel"] == n
    assert weight_summary["target_numel"] == loaded.numel()
    assert eligibility_summary["partial"] is True
    assert eligibility_summary["copied_numel"] == n


@pytest.mark.unit
def test_resume_status_lines_report_missing_and_partial_loads() -> None:
    missing = resume_status_lines(
        {
            "requested": True,
            "loaded": False,
            "reason": "missing",
            "path": "policy.pt",
        }
    )
    loaded = resume_status_lines(
        {
            "requested": True,
            "loaded": True,
            "path": "policy.pt",
            "weights_copied": 4,
            "weights_target": 8,
            "eligibility_copied": 2,
            "eligibility_target": 8,
            "encoder_loaded": True,
            "load_summary": {
                "weights": {"p": {"partial": True}},
                "eligibility": {},
            },
        }
    )

    assert missing == ["Resume requested but not loaded: missing (policy.pt)."]
    assert loaded == [
        "Resume loaded (policy.pt): weights=4/8 (50.0%), "
        "eligibility=2/8 (25.0%), encoder=yes, partial=yes."
    ]


@pytest.mark.unit
def test_consolidation_anchor_protects_loaded_weight_prefix(tmp_path: object) -> None:
    source_net = _small_net()
    source_trainer = _trainer(source_net)
    source_weights = torch.linspace(
        0.0,
        1.0,
        source_trainer.weights_snapshot()["in_to_hidden"].numel(),
        dtype=torch.float32,
    )
    source_topo = source_net.engine.projections["in_to_hidden"].topology  # type: ignore[attr-defined]
    source_topo.weights.copy_(source_weights)

    path = tmp_path / "anchor.pt"  # type: ignore[operator]
    PolicyCheckpoint.save(path, trainer=source_trainer)

    target_net = build_policy_network(
        PolicyNetworkConfig(
            n_input=8,
            n_hidden=12,
            motor_per_button=1,
            input_fanin=0,
            seed=999,
        ),
    )
    target_trainer = OnlineRSTDPTrainer(
        target_net.engine,
        OnlineRSTDPConfig(
            lr=0.0,
            consolidation_strength=0.5,
            plastic_projections=("in_to_hidden",),
        ),
        device="cpu",
        dtype="float32",
    )

    PolicyCheckpoint.load(path, trainer=target_trainer, allow_partial=True)
    target_trainer.anchor_current_weights()
    target_topo = target_net.engine.projections["in_to_hidden"].topology
    target_topo.weights.zero_()

    telemetry = target_trainer.learn(0.0)
    loaded = target_trainer.weights_snapshot()["in_to_hidden"].reshape(-1)
    n = source_weights.numel()
    assert telemetry["consolidation_norm"] > 0.0
    assert torch.allclose(loaded[:n], source_weights * 0.5)
    assert torch.allclose(loaded[n:], torch.zeros_like(loaded[n:]))


@pytest.mark.unit
def test_checkpoint_round_trips_encoder_state(tmp_path: object) -> None:
    net = _small_net()
    trainer = _trainer(net)

    class _FakeEncoder:
        def __init__(self) -> None:
            self.state: dict[str, torch.Tensor] = {"k": torch.tensor([1.0, 2.0, 3.0])}
            self.loaded: dict[str, torch.Tensor] | None = None

        def state_dict(self) -> dict[str, torch.Tensor]:
            return self.state

        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            self.loaded = state

    path = tmp_path / "enc.pt"  # type: ignore[operator]
    PolicyCheckpoint.save(path, trainer=trainer, encoder=_FakeEncoder())

    target = _FakeEncoder()
    PolicyCheckpoint.load(path, trainer=trainer, encoder=target)
    assert target.loaded is not None
    assert torch.allclose(target.loaded["k"], torch.tensor([1.0, 2.0, 3.0]))


#


def _run_offline_task(
    *, stop_after: int | None = None, frames: int = 10,
) -> tuple[object, list[str]]:
    topics: list[str] = []

    class _Collector:
        enabled = True

        def on_event(self, event: object) -> None:
            topics.append(event.topic.value)  # type: ignore[attr-defined]

        def reset(self) -> None: ...
        def snapshot(self) -> dict[str, object]:
            return {}

    bus = EventBus()
    collector = _Collector()
    for topic in EventTopic:
        bus.subscribe(topic, collector)

    calls = {"n": 0}

    def stop_check() -> bool:
        calls["n"] += 1
        return stop_after is not None and calls["n"] > stop_after

    cfg = GameTrainingConfig(
        preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
        n_hidden=24, motor_per_button=2, input_fanin=12,
        rstdp=OnlineRSTDPConfig(lr=1e-2),
        decide_ticks=6, max_episodes=1, frames_per_episode=frames,
        telemetry_every=1, seed=3,
    )
    client = ScriptedGameClient(width=80, height=70, channels=3, max_steps=frames)
    task = GameTrainingTask(cfg, event_bus=bus, stop_check=stop_check, client=client)
    result = task.run()
    return result, topics


@pytest.mark.unit
def test_game_training_task_runs_offline() -> None:
    result, topics = _run_offline_task(frames=8)
    assert result.frames == 8  # type: ignore[attr-defined]
    assert result.stopped is False  # type: ignore[attr-defined]
    assert {"run_start", "topology", "scalar", "training_end", "run_end"} <= set(topics)
    assert topics.count("scalar") == 8


@pytest.mark.unit
def test_game_training_emits_action_energy_metrics() -> None:
    events: list[object] = []

    class _Collector:
        enabled = True

        def on_event(self, event: object) -> None:
            events.append(event)

        def reset(self) -> None: ...
        def snapshot(self) -> dict[str, object]:
            return {}

    bus = EventBus()
    collector = _Collector()
    for topic in EventTopic:
        bus.subscribe(topic, collector)

    cfg = GameTrainingConfig(
        preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
        n_hidden=16,
        motor_per_button=1,
        input_fanin=8,
        decide_ticks=3,
        max_episodes=1,
        frames_per_episode=3,
        telemetry_every=1,
        action_energy=ActionEnergyConfig(enabled=True),
        seed=9,
    )
    task = GameTrainingTask(
        cfg,
        event_bus=bus,
        client=ScriptedGameClient(width=80, height=70, channels=3, max_steps=3),
    )
    task.run()

    scalars = [event for event in events if event.topic.value == "scalar"]  # type: ignore[attr-defined]
    trials = [event for event in events if event.topic.value == "training_trial"]  # type: ignore[attr-defined]
    assert scalars
    assert trials
    assert "action_energy" in scalars[0].data  # type: ignore[attr-defined]
    assert "action_up" in scalars[0].data  # type: ignore[attr-defined]
    assert "reward_action_energy_sum" in trials[0].data  # type: ignore[attr-defined]
    assert "action_button_mean" in trials[0].data  # type: ignore[attr-defined]
    assert "action_up_frac" in trials[0].data  # type: ignore[attr-defined]


@pytest.mark.unit
def test_game_training_task_honours_stop() -> None:
    result, _topics = _run_offline_task(stop_after=3, frames=100)
    assert result.stopped is True  # type: ignore[attr-defined]
    assert result.frames < 100  # type: ignore[attr-defined]


@pytest.mark.unit
def test_game_training_requires_client() -> None:
    with pytest.raises(ValueError, match="client"):
        GameTrainingTask(GameTrainingConfig()).run()


#


class _RecordingClient(ScriptedGameClient):
    """Scripted client that records the savestates queued by the curriculum."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.queued: list[str | None] = []

    def queue_savestate(self, path: str | None) -> None:
        self.queued.append(path)


class _TerminateAfter:
    """Episode manager that ends each episode after a fixed number of frames."""

    def __init__(self, frames: int) -> None:
        self._frames = frames
        self._calls = 0

    def begin_episode(self) -> None:
        self._calls = 0

    def should_end(
        self, before: GameObservation, after: GameObservation,
    ) -> EpisodeDecision:
        del before, after
        self._calls += 1
        return EpisodeDecision(terminated=self._calls >= self._frames, reason="test")


@pytest.mark.unit
def test_task_queues_curriculum_savestate_and_terminates_per_episode() -> None:
    from neuroforge.environments.games.smb3.curriculum import SMB3Curriculum

    client = _RecordingClient(width=80, height=70, channels=3)
    cfg = GameTrainingConfig(
        preprocess=FramePreprocessConfig(out_h=10, out_w=10, motion=False),
        n_hidden=16, motor_per_button=1, input_fanin=8,
        decide_ticks=4, max_episodes=2, frames_per_episode=50, telemetry_every=0, seed=5,
    )
    task = GameTrainingTask(
        cfg,
        client=client,
        episode_manager=_TerminateAfter(3),
        curriculum=SMB3Curriculum(("lvl1.State",)),
    )

    result = task.run()

    assert client.queued == ["lvl1.State", "lvl1.State"]  # queued before each episode
    assert result.episodes == 2
    assert result.frames == 6


@pytest.mark.unit
def test_resume_loads_checkpoint_before_training(tmp_path: Path) -> None:
    ckpt = tmp_path / "policy.pt"
    common = {
        "preprocess": FramePreprocessConfig(out_h=10, out_w=10, motion=False),
        "n_hidden": 16, "motor_per_button": 1, "input_fanin": 8,
        "decide_ticks": 3, "max_episodes": 1, "frames_per_episode": 5,
        "telemetry_every": 0, "seed": 4,
    }
    # First run trains and checkpoints.
    first = GameTrainingTask(
        GameTrainingConfig(checkpoint_path=str(ckpt), **common),  # type: ignore[arg-type]
        client=ScriptedGameClient(width=80, height=70, channels=3, max_steps=5),
    )
    first.run()
    assert ckpt.exists()

    # Second run resumes: run_start must report resumed=True.
    resumed: list[object] = []
    resume_payloads: list[dict[str, object]] = []

    class _RunStartSpy:
        enabled = True

        def on_event(self, event: object) -> None:
            if event.topic.value == "run_start":  # type: ignore[attr-defined]
                data = event.data  # type: ignore[attr-defined]
                resumed.append(data.get("resumed"))
                resume_payloads.append(data.get("resume"))

        def reset(self) -> None: ...
        def snapshot(self) -> dict[str, object]:
            return {}

    bus = EventBus()
    spy = _RunStartSpy()
    for topic in EventTopic:
        bus.subscribe(topic, spy)
    second = GameTrainingTask(
        GameTrainingConfig(checkpoint_path=str(ckpt), resume=True, **common),  # type: ignore[arg-type]
        event_bus=bus,
        client=ScriptedGameClient(width=80, height=70, channels=3, max_steps=5),
    )
    second.run()
    assert resumed == [True]
    assert resume_payloads[0]["loaded"] is True
    assert resume_payloads[0]["weights_copied"] == resume_payloads[0]["weights_target"]
    assert resume_payloads[0]["eligibility_copied"] == resume_payloads[0]["eligibility_target"]


@pytest.mark.unit
def test_resume_checkpoint_path_loads_without_overwriting_source(tmp_path: Path) -> None:
    source = tmp_path / "source_policy.pt"
    common = {
        "preprocess": FramePreprocessConfig(out_h=10, out_w=10, motion=False),
        "n_hidden": 16, "motor_per_button": 1, "input_fanin": 8,
        "decide_ticks": 3, "max_episodes": 1,
        "telemetry_every": 0, "seed": 4,
    }
    first = GameTrainingTask(
        GameTrainingConfig(
            checkpoint_path=str(source),
            frames_per_episode=2,
            **common,  # type: ignore[arg-type]
        ),
        client=ScriptedGameClient(width=80, height=70, channels=3, max_steps=2),
    )
    first.run()
    before = torch.load(str(source), map_location="cpu", weights_only=False)
    assert before["frame"] == 2

    resumed: list[object] = []

    class _RunStartSpy:
        enabled = True

        def on_event(self, event: object) -> None:
            if event.topic.value == "run_start":  # type: ignore[attr-defined]
                resumed.append(event.data.get("resumed"))  # type: ignore[attr-defined]

        def reset(self) -> None: ...
        def snapshot(self) -> dict[str, object]:
            return {}

    bus = EventBus()
    spy = _RunStartSpy()
    for topic in EventTopic:
        bus.subscribe(topic, spy)

    second = GameTrainingTask(
        GameTrainingConfig(
            checkpoint_path=None,
            resume_checkpoint_path=str(source),
            resume=True,
            frames_per_episode=3,
            **common,  # type: ignore[arg-type]
        ),
        event_bus=bus,
        client=ScriptedGameClient(width=80, height=70, channels=3, max_steps=3),
    )
    second.run()

    after = torch.load(str(source), map_location="cpu", weights_only=False)
    assert resumed == [True]
    assert after["frame"] == before["frame"]
    assert after["episode"] == before["episode"]
