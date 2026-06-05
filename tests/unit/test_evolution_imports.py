"""Tests for Phase 5 evolution package boundaries."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


@pytest.mark.unit
def test_evolution_canonical_imports() -> None:
    from neuroforge.neuroevolution.fitness.evaluators import CallableFitnessEvaluator
    from neuroforge.neuroevolution.fitness.objectives import policy_objective_names
    from neuroforge.neuroevolution.genomes.cppn import CPPN
    from neuroforge.neuroevolution.genomes.graph import GraphGenome
    from neuroforge.neuroevolution.genomes.hyperneat import HyperNEATGenome
    from neuroforge.neuroevolution.genomes.innovations import InnovationRegistry
    from neuroforge.neuroevolution.genomes.policy import PolicyGenome
    from neuroforge.neuroevolution.genomes.substrate import SubstrateConfig
    from neuroforge.neuroevolution.io.checkpoints import BestGenomeCheckpoint
    from neuroforge.neuroevolution.io.genome_codec import decode_genome
    from neuroforge.neuroevolution.search.engine import EvolutionEngine

    assert BestGenomeCheckpoint.__name__ == "BestGenomeCheckpoint"
    assert CallableFitnessEvaluator.__name__ == "CallableFitnessEvaluator"
    assert CPPN.__name__ == "CPPN"
    assert EvolutionEngine.__name__ == "EvolutionEngine"
    assert GraphGenome.__name__ == "GraphGenome"
    assert HyperNEATGenome.__name__ == "HyperNEATGenome"
    assert InnovationRegistry.__name__ == "InnovationRegistry"
    assert PolicyGenome.__name__ == "PolicyGenome"
    assert SubstrateConfig.__name__ == "SubstrateConfig"
    assert decode_genome.__name__ == "decode_genome"
    assert policy_objective_names() == ("proxy_policy_gene_target",)


@pytest.mark.unit
def test_legacy_evolution_wrapper_modules_are_not_present() -> None:
    legacy_modules = [
        "neuroforge.applications.evolution",
        "neuroforge.neuroevolution._serde",
        "neuroforge.neuroevolution.checkpoints",
        "neuroforge.neuroevolution.cppn",
        "neuroforge.neuroevolution.engine",
        "neuroforge.neuroevolution.evaluators",
        "neuroforge.neuroevolution.game_evaluators",
        "neuroforge.neuroevolution.genome",
        "neuroforge.neuroevolution.genome_codec",
        "neuroforge.neuroevolution.graph_genome",
        "neuroforge.neuroevolution.hyperneat_genome",
        "neuroforge.neuroevolution.innovations",
        "neuroforge.neuroevolution.neat_ops",
        "neuroforge.neuroevolution.objectives",
        "neuroforge.neuroevolution.substrate",
    ]

    missing = [name for name in legacy_modules if _module_missing(name)]

    assert missing == legacy_modules


@pytest.mark.unit
def test_neuroevolution_does_not_import_game_or_interface_layers() -> None:
    disallowed_prefixes = (
        "neuroforge.dashboard",
        "neuroforge.environments.games.clients.bizhawk",
        "neuroforge.environments.games.smb3",
        "neuroforge.interfaces",
        "neuroforge.observability",
    )
    root = _repo_root() / "src" / "neuroforge" / "neuroevolution"
    offenders: list[str] = []

    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
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
