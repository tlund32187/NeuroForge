# pyright: basic
"""Policy genome for SMB3-oriented neuroevolution.

The first evolution slice searches policy-network structure and learning
hyperparameters that already exist in the live SMB3 training stack. A genome is
therefore a compact, JSON-safe parameterisation of the phenotype built by
``GameTrainingTask`` / ``build_policy_network``.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Any, Literal, cast

from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec

__all__ = ["Gene", "GeneDef", "PolicyGenome", "policy_gene_defs"]

GeneKind = Literal["int", "float", "bool"]


@dataclass(frozen=True, slots=True)
class GeneDef:
    """Definition and mutation bounds for one policy gene."""

    innovation: int
    key: str
    kind: GeneKind
    default: int | float | bool
    minimum: float = 0.0
    maximum: float = 1.0
    sigma: float = 1.0

    def random_value(self, rng: random.Random) -> int | float | bool:
        """Sample a valid value."""
        if self.kind == "bool":
            return rng.random() < 0.5
        if self.kind == "int":
            return rng.randint(int(self.minimum), int(self.maximum))
        return rng.uniform(self.minimum, self.maximum)

    def mutate(
        self, value: int | float | bool, rng: random.Random, *, power: float,
    ) -> int | float | bool:
        """Perturb a value while preserving bounds and type."""
        if self.kind == "bool":
            return not bool(value)
        span = max(1e-12, self.maximum - self.minimum)
        delta = rng.gauss(0.0, self.sigma * power)
        if self.kind == "int":
            raw = int(round(float(value) + delta))
            return max(int(self.minimum), min(int(self.maximum), raw))
        raw_float = float(value) + delta
        # Learning rates are tiny; multiplicative mutation is more useful there.
        if self.key == "lr":
            log_value = math.log(max(float(value), self.minimum))
            log_delta = rng.gauss(0.0, power)
            raw_float = math.exp(log_value + log_delta)
        clipped = max(self.minimum, min(self.maximum, raw_float))
        if span > 1.0:
            return float(clipped)
        return float(clipped)

    def normalised_distance(self, a: int | float | bool, b: int | float | bool) -> float:
        """Return a [0, 1]-ish distance contribution for this gene."""
        if self.kind == "bool":
            return 0.0 if bool(a) == bool(b) else 1.0
        span = max(1e-12, self.maximum - self.minimum)
        return abs(float(a) - float(b)) / span


@dataclass(frozen=True, slots=True)
class Gene:
    """One innovation-numbered gene value."""

    innovation: int
    key: str
    value: int | float | bool


def policy_gene_defs() -> tuple[GeneDef, ...]:
    """Return the ordered gene catalogue for the current SMB3 policy genome."""
    return (
        GeneDef(1, "n_hidden", "int", 128, 32, 256, 24),
        GeneDef(2, "motor_per_button", "int", 4, 1, 8, 1),
        GeneDef(3, "input_fanin", "int", 64, 0, 160, 24),
        GeneDef(4, "recurrent_hidden", "bool", False),
        GeneDef(5, "init_scale", "float", 0.5, 0.05, 1.5, 0.12),
        GeneDef(6, "tau_mem", "float", 5e-3, 2e-3, 20e-3, 1.5e-3),
        GeneDef(7, "decide_ticks", "int", 12, 4, 28, 3),
        GeneDef(8, "noise_amp", "float", 0.4, 0.0, 1.0, 0.08),
        GeneDef(9, "commit_frames", "int", 8, 0, 20, 2),
        GeneDef(10, "lr", "float", 3e-4, 1e-5, 3e-3, 2e-4),
        GeneDef(11, "tau_e", "float", 80e-3, 20e-3, 250e-3, 18e-3),
        GeneDef(12, "reward_scale", "float", 0.05, 0.005, 0.2, 0.02),
    )


_GENE_DEFS: tuple[GeneDef, ...] = policy_gene_defs()
_DEFS_BY_KEY: dict[str, GeneDef] = {definition.key: definition for definition in _GENE_DEFS}


@dataclass(frozen=True, slots=True)
class PolicyGenome:
    """Genome that can instantiate the existing SMB3 spiking policy stack."""

    id: str
    generation: int
    genes: tuple[Gene, ...]
    parent_ids: tuple[str, ...] = ()

    @classmethod
    def seed(
        cls,
        genome_id: str,
        *,
        generation: int = 0,
        rng: random.Random | None = None,
        randomise: bool = True,
    ) -> PolicyGenome:
        """Create a seeded genome from defaults or a random sample."""
        local_rng = rng or random.Random()
        genes = tuple(
            Gene(
                definition.innovation,
                definition.key,
                definition.random_value(local_rng) if randomise else definition.default,
            )
            for definition in _GENE_DEFS
        )
        return cls(id=genome_id, generation=generation, genes=genes)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyGenome:
        """Decode a genome from :meth:`to_dict` output."""
        genes_raw = payload.get("genes", [])
        if not isinstance(genes_raw, list):
            msg = "PolicyGenome payload 'genes' must be a list"
            raise ValueError(msg)
        genes: list[Gene] = []
        for item in genes_raw:
            if not isinstance(item, dict):
                msg = "PolicyGenome gene entries must be objects"
                raise ValueError(msg)
            genes.append(
                Gene(
                    innovation=int(item["innovation"]),
                    key=str(item["key"]),
                    value=cast("int | float | bool", item["value"]),
                )
            )
        parent_raw = payload.get("parent_ids", [])
        parent_ids = tuple(str(pid) for pid in parent_raw) if isinstance(parent_raw, list) else ()
        return cls(
            id=str(payload["id"]),
            generation=int(payload["generation"]),
            genes=tuple(genes),
            parent_ids=parent_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "type": "policy",
            "id": self.id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "genes": [dataclasses.asdict(gene) for gene in self.genes],
        }

    def value(self, key: str) -> int | float | bool:
        """Return one gene value by key."""
        for gene in self.genes:
            if gene.key == key:
                return gene.value
        msg = f"gene {key!r} not found"
        raise KeyError(msg)

    def mutate(
        self,
        *,
        child_id: str,
        generation: int,
        rng: random.Random,
        rate: float,
        power: float = 1.0,
    ) -> PolicyGenome:
        """Return a mutated child genome."""
        next_genes: list[Gene] = []
        changed = False
        forced_index = rng.randrange(len(self.genes)) if self.genes else -1
        for idx, gene in enumerate(self.genes):
            definition = _DEFS_BY_KEY[gene.key]
            should_mutate = rng.random() < rate or idx == forced_index
            if should_mutate:
                value = definition.mutate(gene.value, rng, power=power)
                changed = changed or value != gene.value
            else:
                value = gene.value
            next_genes.append(Gene(gene.innovation, gene.key, value))
        if not changed and next_genes:
            gene = next_genes[forced_index]
            definition = _DEFS_BY_KEY[gene.key]
            value = definition.mutate(gene.value, rng, power=max(1.0, power))
            next_genes[forced_index] = Gene(gene.innovation, gene.key, value)
        return PolicyGenome(
            id=child_id,
            generation=generation,
            genes=tuple(next_genes),
            parent_ids=(self.id,),
        )

    def crossover(
        self,
        other: PolicyGenome,
        *,
        child_id: str,
        generation: int,
        rng: random.Random,
    ) -> PolicyGenome:
        """Uniform crossover by matching innovation numbers."""
        other_by_innovation = {gene.innovation: gene for gene in other.genes}
        child_genes: list[Gene] = []
        for gene in self.genes:
            mate = other_by_innovation.get(gene.innovation)
            chosen = mate if mate is not None and rng.random() < 0.5 else gene
            child_genes.append(Gene(chosen.innovation, chosen.key, chosen.value))
        return PolicyGenome(
            id=child_id,
            generation=generation,
            genes=tuple(child_genes),
            parent_ids=(self.id, other.id),
        )

    def distance(self, other: PolicyGenome) -> float:
        """Compatibility distance for simple speciation."""
        other_by_key = {gene.key: gene for gene in other.genes}
        total = 0.0
        compared = 0
        for gene in self.genes:
            mate = other_by_key.get(gene.key)
            if mate is None:
                total += 1.0
                compared += 1
                continue
            definition = _DEFS_BY_KEY[gene.key]
            total += definition.normalised_distance(gene.value, mate.value)
            compared += 1
        return total / max(1, compared)

    def to_policy_network_config(
        self,
        *,
        n_input: int,
        seed: int,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> Any:
        """Build a :class:`PolicyNetworkConfig` for this genome."""
        from neuroforge.game.policies.network import PolicyNetworkConfig

        return PolicyNetworkConfig(
            n_input=n_input,
            n_hidden=int(self.value("n_hidden")),
            motor_per_button=int(self.value("motor_per_button")),
            input_fanin=int(self.value("input_fanin")),
            recurrent_hidden=bool(self.value("recurrent_hidden")),
            init_scale=float(self.value("init_scale")),
            tau_mem=float(self.value("tau_mem")),
            seed=seed,
            device=device,
            dtype=dtype,
        )

    def to_game_training_config(self, *, base: Any | None = None, seed: int) -> Any:
        """Build a :class:`GameTrainingConfig` for this genome."""
        from neuroforge.learning.online_rstdp import OnlineRSTDPConfig
        from neuroforge.tasks.game_training import GameTrainingConfig

        cfg = base or GameTrainingConfig()
        rstdp = dataclasses.replace(
            cfg.rstdp if hasattr(cfg, "rstdp") else OnlineRSTDPConfig(),
            lr=float(self.value("lr")),
            tau_e=float(self.value("tau_e")),
            reward_scale=float(self.value("reward_scale")),
        )
        return dataclasses.replace(
            cfg,
            n_hidden=int(self.value("n_hidden")),
            motor_per_button=int(self.value("motor_per_button")),
            input_fanin=int(self.value("input_fanin")),
            recurrent_hidden=bool(self.value("recurrent_hidden")),
            init_scale=float(self.value("init_scale")),
            tau_mem=float(self.value("tau_mem")),
            decide_ticks=int(self.value("decide_ticks")),
            noise_amp=float(self.value("noise_amp")),
            commit_frames=int(self.value("commit_frames")),
            rstdp=rstdp,
            seed=seed,
        )

    def to_network_spec(self, *, n_input: int) -> NetworkSpec:
        """Return a declarative spec summary for dashboards/checkpoints.

        The live phenotype is built by ``build_policy_network`` because Dale
        signs are tensor-valued. This spec mirrors the evolved macro-structure
        and stores the policy-builder details in metadata.
        """
        n_hidden = int(self.value("n_hidden"))
        n_motor = int(self.value("motor_per_button")) * 8
        input_fanin = int(self.value("input_fanin"))
        input_topology: dict[str, Any]
        if 0 < input_fanin < n_input:
            input_topology = {
                "type": "sparse_fanin",
                "fanin": input_fanin,
                "init": "uniform",
                "low": 0.0,
                "high": float(self.value("init_scale")),
            }
        else:
            input_topology = {
                "type": "dense",
                "init": "uniform",
                "low": 0.0,
                "high": float(self.value("init_scale")),
            }
        projections = [
            ProjectionSpec("in_to_hidden", "input", "hidden", "static", topology=input_topology),
            ProjectionSpec(
                "hidden_to_motor",
                "hidden",
                "motor",
                "static",
                topology={
                    "type": "dense",
                    "init": "uniform",
                    "low": 0.0,
                    "high": float(self.value("init_scale")),
                },
            ),
        ]
        if bool(self.value("recurrent_hidden")):
            projections.append(
                ProjectionSpec(
                    "hidden_to_hidden",
                    "hidden",
                    "hidden",
                    "static",
                    topology={
                        "type": "dense",
                        "init": "uniform",
                        "low": 0.0,
                        "high": float(self.value("init_scale")),
                    },
                )
            )
        return NetworkSpec(
            populations=[
                PopulationSpec("input", n_input, "lif", {"tau_mem": float(self.value("tau_mem"))}),
                PopulationSpec(
                    "hidden", n_hidden, "lif", {"tau_mem": float(self.value("tau_mem"))},
                ),
                PopulationSpec("motor", n_motor, "lif", {"tau_mem": float(self.value("tau_mem"))}),
            ],
            projections=projections,
            metadata={
                "genome_id": self.id,
                "generation": self.generation,
                "phenotype_builder": "neuroforge.game.policies.network.build_policy_network",
                "decide_ticks": int(self.value("decide_ticks")),
                "noise_amp": float(self.value("noise_amp")),
                "commit_frames": int(self.value("commit_frames")),
            },
        )
