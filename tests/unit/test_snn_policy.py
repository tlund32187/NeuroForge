"""Unit tests for the spiking game policy (Phase 2).

Covers frame preprocessing, policy-network construction, stateful inference
(persistence + readout), multi-label action decoding with d-pad conflict
resolution, and the full policy driving the vision-only game loop.
"""

from __future__ import annotations

import pytest
import torch

from neuroforge.contracts.game import (
    ControllerAction,
    GameObservation,
    IVisionGamePolicy,
    ScreenFrame,
)
from neuroforge.game.clients.scripted import ScriptedGameClient
from neuroforge.game.loop import VisionOnlyGameLoop
from neuroforge.game.policies import (
    ActionDecodeConfig,
    ActionDecoder,
    CoreEnginePolicyEngine,
    FramePreprocessConfig,
    FramePreprocessor,
    PolicyNetworkConfig,
    build_policy_network,
    build_snn_game_policy,
)


def _frame(fill: int, *, w: int = 64, h: int = 56, c: int = 3) -> ScreenFrame:
    return ScreenFrame(width=w, height=h, channels=c, data=bytes([fill % 256]) * (w * h * c))


# ── frame preprocessing ────────────────────────────────────────────────


@pytest.mark.unit
def test_preprocessor_input_size_and_motion() -> None:
    gray = FramePreprocessor(FramePreprocessConfig(out_h=4, out_w=5, motion=False))
    assert gray.input_size == 20
    motion = FramePreprocessor(FramePreprocessConfig(out_h=4, out_w=5, motion=True))
    assert motion.input_size == 40


@pytest.mark.unit
def test_preprocessor_drive_shape_and_amplitude() -> None:
    pre = FramePreprocessor(FramePreprocessConfig(out_h=4, out_w=4, amplitude=10.0, motion=False))
    drive = pre.to_drive(_frame(255))
    assert tuple(drive.shape) == (16,)
    assert float(drive.max()) == pytest.approx(10.0, abs=0.5)  # full-brightness ~ amplitude
    assert float(pre.to_drive(_frame(0)).max()) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.unit
def test_preprocessor_motion_first_frame_is_zero() -> None:
    pre = FramePreprocessor(FramePreprocessConfig(out_h=3, out_w=3, motion=True))
    drive = pre.to_drive(_frame(200))
    assert tuple(drive.shape) == (18,)
    assert float(drive[9:].abs().max()) == pytest.approx(0.0, abs=1e-4)  # diff half = 0


# ── network construction ─────────────────────────────────────────────────


@pytest.mark.unit
def test_build_policy_network_shapes() -> None:
    net = build_policy_network(
        PolicyNetworkConfig(n_input=20, n_hidden=16, n_inhibitory_hidden=4, motor_per_button=2),
    )
    assert net.n_motor == 16  # 8 buttons * 2
    assert set(net.engine.populations) == {"input", "hidden", "motor"}
    assert net.engine.populations["input"].n == 20
    assert net.engine.populations["motor"].n == 16
    assert "in_to_hidden" in net.plastic_projections
    assert "hidden_to_motor" in net.plastic_projections


@pytest.mark.unit
def test_build_policy_network_recurrent_adds_projection() -> None:
    net = build_policy_network(
        PolicyNetworkConfig(n_input=10, n_hidden=8, motor_per_button=1, recurrent_hidden=True),
    )
    assert "hidden_to_hidden" in net.plastic_projections
    assert "hidden_to_hidden" in net.engine.projections


# ── stateful inference ───────────────────────────────────────────────────


@pytest.mark.unit
def test_decide_returns_valid_rates_and_persists_state() -> None:
    net = build_policy_network(PolicyNetworkConfig(n_input=20, n_hidden=16, motor_per_button=2))
    engine = CoreEnginePolicyEngine(
        net.engine, motor_pop="motor", motor_per_button=2, n_buttons=8,
    )
    drive = torch.full((20,), 10.0)
    decision = engine.decide(drive, ticks=10)
    assert tuple(decision.motor_rates.shape) == (8,)
    assert bool((decision.motor_rates >= 0).all())
    assert bool((decision.motor_rates <= 1).all())

    # State persisted (hidden charged), and reset returns it to rest.
    hidden_v = net.engine.populations["hidden"].state["v"]
    assert bool((hidden_v.abs() > 0).any())
    engine.reset()
    assert bool((net.engine.populations["hidden"].state["v"] == 0).all())


@pytest.mark.unit
def test_decide_rejects_nonpositive_ticks() -> None:
    net = build_policy_network(PolicyNetworkConfig(n_input=8, n_hidden=8, motor_per_button=1))
    engine = CoreEnginePolicyEngine(net.engine, motor_pop="motor", motor_per_button=1, n_buttons=8)
    with pytest.raises(ValueError, match="ticks"):
        engine.decide(torch.zeros(8), ticks=0)


# ── action decoding ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_decoder_multilabel_and_dpad_conflict() -> None:
    decoder = ActionDecoder(ActionDecodeConfig(mode="threshold", threshold=0.5))
    # order: Up, Down, Left, Right, A, B, Start, Select
    rates = torch.tensor([0.1, 0.1, 0.9, 0.8, 0.6, 0.1, 0.9, 0.9])
    action = decoder.decode(rates)
    assert action.left and not action.right       # Left(0.9) beats Right(0.8)
    assert not action.up and not action.down      # both below threshold
    assert action.a and not action.b              # multi-label: A on, B off
    assert not action.start and not action.select  # masked by default


@pytest.mark.unit
def test_decoder_allows_start_when_configured() -> None:
    decoder = ActionDecoder(ActionDecodeConfig(mode="threshold", threshold=0.5, allow_start=True))
    action = decoder.decode(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]))
    assert action.start


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["threshold", "bernoulli", "epsilon"])
def test_decoder_modes_produce_valid_actions(mode: str) -> None:
    decoder = ActionDecoder(ActionDecodeConfig(mode=mode, epsilon=0.5), seed=3)
    # Should never raise (d-pad conflicts resolved) regardless of mode/rates.
    action = decoder.decode(torch.tensor([0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5]))
    assert isinstance(action, ControllerAction)


@pytest.mark.unit
def test_threshold_dpad_tiebreak_is_deterministic() -> None:
    # Deterministic eval: the higher-rate direction always wins in threshold mode.
    decoder = ActionDecoder(ActionDecodeConfig(mode="threshold", threshold=0.3))
    rates = torch.tensor([0.0, 0.0, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0])  # Left 0.9 > Right 0.5
    for _ in range(20):
        action = decoder.decode(rates)
        assert action.left and not action.right


@pytest.mark.unit
def test_stochastic_dpad_tiebreak_escapes_a_fixed_bias() -> None:
    # The "only ever goes left" fix: even with Left firing far more than Right,
    # bernoulli decoding must pick Right at least sometimes so the other direction
    # can be explored and its reward discovered.
    decoder = ActionDecoder(
        ActionDecodeConfig(mode="bernoulli", threshold=0.25, temperature=0.3,
                           dpad_explore_floor=0.1),
        seed=0,
    )
    rates = torch.tensor([0.0, 0.0, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0])  # strong Left bias
    rights = sum(decoder.decode(rates).right for _ in range(400))
    assert rights > 0  # Right is reachable despite the Left bias


# ── full policy through the loop ─────────────────────────────────────────


@pytest.mark.unit
def test_snn_policy_drives_loop() -> None:
    policy = build_snn_game_policy(
        preprocess=FramePreprocessConfig(out_h=8, out_w=8, motion=False),
        network=PolicyNetworkConfig(n_input=1, n_hidden=16, motor_per_button=2, input_fanin=8),
        decide_ticks=8,
        seed=2,
    )
    assert isinstance(policy, IVisionGamePolicy)

    policy.begin_episode()
    client = ScriptedGameClient(width=64, height=56, channels=3, max_steps=3)
    loop = VisionOnlyGameLoop(client=client, policy=policy)
    transitions = list(loop.run(max_steps=10))

    assert len(transitions) == 3
    assert all(isinstance(t.action, ControllerAction) for t in transitions)
    assert policy.last_decision is not None
    assert tuple(policy.last_decision.motor_rates.shape) == (8,)


@pytest.mark.unit
def test_build_snn_policy_forces_input_size_match() -> None:
    # Network n_input is deliberately wrong; the builder must override it to the
    # preprocessor's output size, so construction succeeds and runs.
    policy = build_snn_game_policy(
        preprocess=FramePreprocessConfig(out_h=6, out_w=6, motion=True),
        network=PolicyNetworkConfig(n_input=999, n_hidden=8, motor_per_button=1),
        decide_ticks=6,
    )
    action = policy.act(GameObservation(step=0, t=0.0, frame=_frame(120)))
    assert isinstance(action, ControllerAction)
