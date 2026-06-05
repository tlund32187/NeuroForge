"""Registry of game-client constructors.

Game clients are a higher-level, emulator-facing concern than the brain
components in :class:`~neuroforge.construction.hub.FactoryHub`, so they get their
own registry here (separation of concerns) while reusing the same generic
:class:`~neuroforge.construction.registry.Registry`. A future game task resolves a
client by key and depends only on the :class:`IGameClient` protocol.

Usage::

    client = GAME_CLIENTS.create("bizhawk", config=BizHawkClientConfig(...))
    client = GAME_CLIENTS.create("scripted", width=84, height=84, channels=1)
"""

from __future__ import annotations

from neuroforge.construction.registry import Registry

__all__ = ["GAME_CLIENTS", "build_game_client_registry"]


def build_game_client_registry() -> Registry:
    """Create a registry populated with the built-in game clients."""
    from neuroforge.environments.games.clients.bizhawk.client import BizHawkClient
    from neuroforge.environments.games.clients.scripted import ReplayGameClient, ScriptedGameClient

    registry: Registry = Registry("game_clients")
    registry.register("bizhawk", BizHawkClient)
    registry.register("scripted", ScriptedGameClient)
    registry.register("replay", ReplayGameClient)
    return registry


GAME_CLIENTS: Registry = build_game_client_registry()
