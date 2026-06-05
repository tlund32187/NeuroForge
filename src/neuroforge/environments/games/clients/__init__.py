"""Emulator client implementations for the vision-only game seam.

This package provides concrete :class:`~neuroforge.contracts.applications.games.IGameClient`
implementations and the wire protocol that connects a NeuroForge brain to a
real emulator (BizHawk) or to deterministic in-memory fakes for testing.

The public surface intentionally exposes only frame pixels, controller
actions, and frame-derived observations — never emulator RAM — preserving the
vision-only invariant of :mod:
euroforge.contracts.game`.
"""

from neuroforge.environments.games.clients.bizhawk.client import (
    BizHawkClient,
    BizHawkClientConfig,
)
from neuroforge.environments.games.clients.bizhawk.errors import (
    BizHawkConnectionError,
    BizHawkProtocolError,
    BizHawkStateError,
    BridgeError,
)
from neuroforge.environments.games.clients.scripted import (
    ActionProgressGameClient,
    ReplayGameClient,
    ScriptedGameClient,
)

__all__ = [
    "ActionProgressGameClient",
    "BizHawkClient",
    "BizHawkClientConfig",
    "BridgeError",
    "BizHawkConnectionError",
    "BizHawkProtocolError",
    "BizHawkStateError",
    "ReplayGameClient",
    "ScriptedGameClient",
]
