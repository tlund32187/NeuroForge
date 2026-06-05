"""Agent/game architecture boundary tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_files(*roots: Path) -> list[Path]:
    return [
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


@pytest.mark.unit
def test_agent_game_architecture_doc_exists() -> None:
    path = _repo_root() / "docs" / "architecture" / "agent_game_architecture.md"
    text = path.read_text(encoding="utf-8")

    required_terms = [
        "neuroforge.agents",
        "neuroforge.neuroevolution",
        "environments.games.clients.bizhawk",
        "environments.games.smb3",
        "applications.smb3",
        "No forwarding import wrappers",
    ]
    missing = [term for term in required_terms if term not in text]
    assert missing == []


@pytest.mark.unit
def test_phase_eight_repo_level_assets_exist() -> None:
    root = _repo_root()
    paths = [
        root / "configs" / "smb3" / "default.toml",
        root / "configs" / "agents" / "default.toml",
        root / "configs" / "neuroevolution" / "default.toml",
        root / "scripts" / "bizhawk" / "neuroforge_bridge.lua",
        root / "examples" / "smb3" / "train_minimal.py",
        root / "examples" / "custom_agent" / "simple_agent.py",
        root / "examples" / "custom_brain" / "simple_brain.py",
        root / "examples" / "custom_environment" / "simple_environment.py",
        root / "examples" / "custom_neat_genome" / "seed_policy_genome.py",
        root / "examples" / "custom_hyperneat_substrate" / "substrate_config.py",
    ]

    missing = [str(path.relative_to(root)) for path in paths if not path.exists()]

    assert missing == []


@pytest.mark.unit
def test_new_architecture_skeleton_packages_import_without_optional_deps() -> None:
    modules = [
        "neuroforge.agents",
        "neuroforge.agents.base",
        "neuroforge.agents.brain",
        "neuroforge.agents.policies.policy",
        "neuroforge.neuroevolution",
        "neuroforge.environments.games.clients.bizhawk",
        "neuroforge.applications.smb3",
    ]
    for module in modules:
        assert importlib.import_module(module).__name__ == module


@pytest.mark.unit
def test_old_bizhawk_module_paths_are_not_importable() -> None:
    old_modules = [
        "neuroforge.environments.games.clients.bizhawk_client",
        "neuroforge.environments.games.clients.errors",
        "neuroforge.environments.games.clients.file_transport",
        "neuroforge.environments.games.clients.frame_codec",
        "neuroforge.environments.games.clients.launcher",
        "neuroforge.environments.games.clients.protocol",
        "neuroforge.environments.games.clients.screenshot_socket",
        "neuroforge.environments.games.clients.transport",
    ]
    for module in old_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


@pytest.mark.unit
def test_source_does_not_import_deleted_game_paths() -> None:
    old_imports = (
        "neuroforge.environments.games.clients.bizhawk_client",
        "neuroforge.environments.games.clients.errors",
        "neuroforge.environments.games.clients.file_transport",
        "neuroforge.environments.games.clients.frame_codec",
        "neuroforge.environments.games.clients.launcher",
        "neuroforge.environments.games.clients.protocol",
        "neuroforge.environments.games.clients.screenshot_socket",
        "neuroforge.environments.games.clients.transport",
        "neuroforge.environments.games.smb3.action_energy",
        "neuroforge.environments.games.smb3.checkpoint",
        "neuroforge.environments.games.smb3.hud.hud_extractor",
        "neuroforge.environments.games.smb3.loop",
        "neuroforge.environments.games.smb3.policies",
        "neuroforge.environments.games.smb3.policies.action_decode",
        "neuroforge.environments.games.smb3.policies.commitment",
        "neuroforge.environments.games.smb3.policies.encoder",
        "neuroforge.environments.games.smb3.policies.graph_network",
        "neuroforge.environments.games.smb3.policies.hyperneat_network",
        "neuroforge.environments.games.smb3.policies.network",
        "neuroforge.environments.games.smb3.policies.preprocess",
        "neuroforge.environments.games.smb3.policies.snn_policy",
        "neuroforge.environments.games.smb3.policies.stateful_engine",
        "neuroforge.environments.games.smb3.rewards_smb3",
        "neuroforge.environments.games.smb3.smb3_live",
    )
    roots = (_repo_root() / "src", _repo_root() / "scripts", _repo_root() / "tests")
    offenders: list[str] = []
    for path in _python_files(*roots):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            else:
                continue
            offenders.extend(
                f"{path}:{name}" for name in imported if any(name == old for old in old_imports)
            )

    assert offenders == []


@pytest.mark.unit
def test_examples_use_canonical_import_paths() -> None:
    old_prefixes = (
        "neuroforge.evolution",
        "neuroforge.game",
        "neuroforge.vision",
        "neuroforge.encoding",
        "neuroforge.encoders",
        "neuroforge.readout",
        "neuroforge.monitors",
        "neuroforge.tasks",
    )
    offenders: list[str] = []
    for path in _python_files(_repo_root() / "examples"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            else:
                continue
            offenders.extend(
                f"{path}:{name}"
                for name in names
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in old_prefixes)
            )

    assert offenders == []


@pytest.mark.unit
def test_old_smb3_module_paths_are_not_importable() -> None:
    old_modules = [
        "neuroforge.environments.games.smb3.action_energy",
        "neuroforge.environments.games.smb3.checkpoint",
        "neuroforge.environments.games.smb3.hud.hud_extractor",
        "neuroforge.environments.games.smb3.loop",
        "neuroforge.environments.games.smb3.policies",
        "neuroforge.environments.games.smb3.policies.action_decode",
        "neuroforge.environments.games.smb3.policies.commitment",
        "neuroforge.environments.games.smb3.policies.encoder",
        "neuroforge.environments.games.smb3.policies.graph_network",
        "neuroforge.environments.games.smb3.policies.hyperneat_network",
        "neuroforge.environments.games.smb3.policies.network",
        "neuroforge.environments.games.smb3.policies.preprocess",
        "neuroforge.environments.games.smb3.policies.snn_policy",
        "neuroforge.environments.games.smb3.policies.stateful_engine",
        "neuroforge.environments.games.smb3.rewards_smb3",
        "neuroforge.environments.games.smb3.smb3_live",
    ]
    for module in old_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


@pytest.mark.unit
def test_bizhawk_bridge_imports_do_not_depend_on_game_or_app_layers() -> None:
    disallowed_prefixes = (
        "neuroforge.agents",
        "neuroforge.applications",
        "neuroforge.dashboard",
        "neuroforge.environments.games.smb3",
        "neuroforge.interfaces",
        "neuroforge.neuroevolution",
    )
    root = _repo_root() / "src" / "neuroforge" / "environments" / "games" / "clients" / "bizhawk"
    offenders: list[str] = []

    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            else:
                continue
            offenders.extend(
                f"{path}:{name}"
                for name in names
                if any(name.startswith(prefix) for prefix in disallowed_prefixes)
            )

    assert offenders == []
