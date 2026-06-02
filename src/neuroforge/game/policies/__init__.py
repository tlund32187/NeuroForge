"""Spiking policy: frame pixels -> stateful SNN -> controller buttons."""

from neuroforge.game.policies.action_decode import ActionDecodeConfig, ActionDecoder
from neuroforge.game.policies.network import (
    N_BUTTONS,
    PolicyNetwork,
    PolicyNetworkConfig,
    build_policy_network,
)
from neuroforge.game.policies.preprocess import FramePreprocessConfig, FramePreprocessor
from neuroforge.game.policies.snn_policy import SNNGamePolicy, build_snn_game_policy
from neuroforge.game.policies.stateful_engine import (
    CoreEnginePolicyEngine,
    IStatefulPolicyEngine,
    PolicyDecision,
)

__all__ = [
    "ActionDecodeConfig",
    "ActionDecoder",
    "CoreEnginePolicyEngine",
    "FramePreprocessConfig",
    "FramePreprocessor",
    "IStatefulPolicyEngine",
    "N_BUTTONS",
    "PolicyDecision",
    "PolicyNetwork",
    "PolicyNetworkConfig",
    "SNNGamePolicy",
    "build_policy_network",
    "build_snn_game_policy",
]
