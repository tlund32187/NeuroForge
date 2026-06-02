"""Unit tests for the online R-STDP learning loop (Phase 3).

Drives the trainer with controlled pre/post spikes to verify the learning
mechanics directly: reward sign moves weights the right way, Dale-safe clamping
keeps magnitudes non-negative, eligibility bridges an action->reward delay, the
reward shaper computes a clipped RPE, checkpoints round-trip, and the full task
runs offline and honours stop requests.
"""

from __future__ import annotations

import pytest
import torch

from neuroforge.contracts.game import EpisodeDecision, GameObservation
from neuroforge.contracts.monitors import EventTopic
from neuroforge.game.checkpoint import PolicyCheckpoint
from neuroforge.game.clients.scripted import ScriptedGameClient
from neuroforge.game.policies.network import PolicyNetworkConfig, build_policy_network
from neuroforge.game.policies.preprocess import FramePreprocessConfig
from neuroforge.learning.online_rstdp import (
    OnlineRSTDPConfig,
    OnlineRSTDPTrainer,
    RewardShaper,
)
from neuroforge.learning.rstdp import RSTDPRule
from neuroforge.monitors.bus import EventBus
from neuroforge.tasks.game_training import GameTrainingConfig, GameTrainingTask


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


# ── learning mechanics ──────────────────────────────────────────────────


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


# ── checkpointing ─────────────────────────────────────────────────────────


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


# ── full task offline ──────────────────────────────────────────────────────


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
def test_game_training_task_honours_stop() -> None:
    result, _topics = _run_offline_task(stop_after=3, frames=100)
    assert result.stopped is True  # type: ignore[attr-defined]
    assert result.frames < 100  # type: ignore[attr-defined]


@pytest.mark.unit
def test_game_training_requires_client() -> None:
    with pytest.raises(ValueError, match="client"):
        GameTrainingTask(GameTrainingConfig()).run()


# ── Phase 4 wiring: curriculum savestates + vision-derived termination ─────


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
    from neuroforge.game.curriculum import SMB3Curriculum

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
    assert result.episodes == 2  # type: ignore[attr-defined]
    assert result.frames == 6  # type: ignore[attr-defined] - 3 frames/episode via the manager


@pytest.mark.unit
def test_resume_loads_checkpoint_before_training(tmp_path: object) -> None:
    ckpt = tmp_path / "policy.pt"  # type: ignore[operator]
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
    assert ckpt.exists()  # type: ignore[attr-defined]

    # Second run resumes: run_start must report resumed=True.
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
        GameTrainingConfig(checkpoint_path=str(ckpt), resume=True, **common),  # type: ignore[arg-type]
        event_bus=bus,
        client=ScriptedGameClient(width=80, height=70, channels=3, max_steps=5),
    )
    second.run()
    assert resumed == [True]
