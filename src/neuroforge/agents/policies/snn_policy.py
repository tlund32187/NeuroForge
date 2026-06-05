"""``SNNGamePolicy`` — a vision-only game policy backed by a spiking brain.

Thin adapter implementing :class:`~neuroforge.contracts.applications.games.IVisionGamePolicy`:
frame -> :class:`FramePreprocessor` -> :class:`IStatefulPolicyEngine` ->
:class:`ActionDecoder` -> :class:`ControllerAction`. Neural state persists
across ``act`` calls; ``begin_episode`` clears it.

The weights are random until the Phase-3 R-STDP loop trains them, so this drives
the game but does not (yet) play it well.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from neuroforge.agents.actuators.action_decoder import (
    ActionDecodeConfig,
    ActionDecoder,
)
from neuroforge.agents.brains.policy_network import (
    PolicyNetworkConfig,
    build_policy_network,
)
from neuroforge.agents.brains.stateful_engine import (
    CoreEnginePolicyEngine,
    IStatefulPolicyEngine,
    PolicyDecision,
)
from neuroforge.perception.vision.encoding.frame_preprocess import (
    FramePreprocessConfig,
    FramePreprocessor,
)

if TYPE_CHECKING:
    from neuroforge.agents.actuators.action_commitment import TemporalCommitment
    from neuroforge.construction.hub import FactoryHub
    from neuroforge.contracts.applications.games import ControllerAction, GameObservation
    from neuroforge.perception.vision.encoding.frame_encoder import IFrameEncoder

__all__ = ["SNNGamePolicy", "build_snn_game_policy"]


class SNNGamePolicy:
    """Adapt a stateful spiking brain to the game policy protocol."""

    def __init__(
        self,
        *,
        engine: IStatefulPolicyEngine,
        preprocessor: IFrameEncoder,
        decoder: ActionDecoder,
        decide_ticks: int = 12,
        commitment: TemporalCommitment | None = None,
    ) -> None:
        if decide_ticks < 1:
            msg = "decide_ticks must be >= 1"
            raise ValueError(msg)
        self._engine = engine
        self._pre = preprocessor
        self._decoder = decoder
        self._ticks = int(decide_ticks)
        self._commitment = commitment
        self._last_decision: PolicyDecision | None = None

    def act(self, observation: GameObservation) -> ControllerAction:
        drive = self._pre.to_drive(observation.frame)
        decision = self._engine.decide(drive, ticks=self._ticks)
        self._last_decision = decision
        action = self._decoder.decode(decision.motor_rates)
        if self._commitment is not None:
            action = self._commitment.apply(action)
        return action

    def begin_episode(self) -> None:
        """Reset neural + motion state at the start of an episode/life."""
        self._engine.reset()
        self._pre.reset()
        if self._commitment is not None:
            self._commitment.reset()

    @property
    def last_decision(self) -> PolicyDecision | None:
        return self._last_decision


def build_snn_game_policy(
    *,
    preprocess: FramePreprocessConfig | None = None,
    network: PolicyNetworkConfig | None = None,
    decode: ActionDecodeConfig | None = None,
    decide_ticks: int = 12,
    noise_amp: float = 0.0,
    seed: int = 42,
    hub: FactoryHub | None = None,
) -> SNNGamePolicy:
    """Wire a preprocessor, policy network, stateful engine, and decoder together.

    The network's input size is forced to match the preprocessor's output, so
    the two can never disagree.
    """
    preprocessor = FramePreprocessor(preprocess or FramePreprocessConfig())
    net_cfg = network or PolicyNetworkConfig(n_input=preprocessor.input_size, seed=seed)
    net_cfg = dataclasses.replace(net_cfg, n_input=preprocessor.input_size)
    net = build_policy_network(net_cfg, hub=hub)
    engine: Any = CoreEnginePolicyEngine(
        net.engine,
        motor_pop=net.motor_pop,
        motor_per_button=net.motor_per_button,
        n_buttons=net.n_buttons,
        input_pop=net.input_pop,
        noise_amp=noise_amp,
        seed=seed,
    )
    decoder = ActionDecoder(decode or ActionDecodeConfig(), seed=seed)
    return SNNGamePolicy(
        engine=engine, preprocessor=preprocessor, decoder=decoder, decide_ticks=decide_ticks,
    )
