# pyright: basic
"""Reusable fitness objectives for policy-genome evolution."""

from __future__ import annotations

from collections.abc import Callable

from neuroforge.contracts.evolution import FitnessResult
from neuroforge.evolution.genome import PolicyGenome

__all__ = [
    "PolicyObjective",
    "evaluate_proxy_policy_gene_target",
    "get_policy_objective",
    "policy_objective_names",
]

PolicyObjective = Callable[[PolicyGenome], FitnessResult]
_PROXY_POLICY_OBJECTIVE = "proxy_policy_gene_target"
_ALIASES: dict[str, str] = {
    _PROXY_POLICY_OBJECTIVE: _PROXY_POLICY_OBJECTIVE,
    "proxy": _PROXY_POLICY_OBJECTIVE,
}


def policy_objective_names() -> tuple[str, ...]:
    """Return canonical built-in objective names."""
    return (_PROXY_POLICY_OBJECTIVE,)


def get_policy_objective(name: str) -> PolicyObjective:
    """Look up a built-in objective by name."""
    key = _ALIASES.get(name.strip().lower())
    if key == _PROXY_POLICY_OBJECTIVE:
        return evaluate_proxy_policy_gene_target
    known = ", ".join(policy_objective_names())
    msg = f"Unknown policy objective {name!r}; expected one of: {known}"
    raise ValueError(msg)


def evaluate_proxy_policy_gene_target(genome: PolicyGenome) -> FitnessResult:
    """Score a genome against a fast deterministic SMB3-policy proxy target.

    This is a bring-up objective, not a gameplay score. It exercises selection,
    mutation, speciation, checkpointing, and artifact publishing in milliseconds
    so the evolution pipeline can be tested before attaching live BizHawk runs.
    """
    hidden = _float_gene(genome, "n_hidden")
    motors = _float_gene(genome, "motor_per_button")
    fanin = _float_gene(genome, "input_fanin")
    init_scale = _float_gene(genome, "init_scale")
    tau_mem = _float_gene(genome, "tau_mem")
    decide_ticks = _float_gene(genome, "decide_ticks")
    noise_amp = _float_gene(genome, "noise_amp")
    commit_frames = _float_gene(genome, "commit_frames")
    lr = _float_gene(genome, "lr")
    tau_e = _float_gene(genome, "tau_e")
    reward_scale = _float_gene(genome, "reward_scale")
    hidden_layers = _float_gene(genome, "n_hidden_layers")
    hidden_fanin = _float_gene(genome, "hidden_fanin")
    recurrent = bool(genome.value("recurrent_hidden"))
    input_skip = bool(genome.value("input_to_motor_skip"))

    target_cost = (
        abs(hidden - 160.0) / 64.0
        + abs(motors - 5.0) / 3.0
        + abs(fanin - 72.0) / 72.0
        + abs(init_scale - 0.55) / 0.55
        + abs(tau_mem - 0.006) / 0.006
        + abs(decide_ticks - 10.0) / 10.0
        + abs(noise_amp - 0.35) / 0.35
        + abs(commit_frames - 4.0) / 4.0
        + abs(lr - 4e-4) / 4e-4
        + abs(tau_e - 0.10) / 0.10
        + abs(reward_scale - 0.06) / 0.06
        + abs(hidden_layers - 2.0) / 2.0
        + abs(hidden_fanin - 64.0) / 96.0
    )
    recurrent_bonus = 0.35 if recurrent else 0.0
    skip_bonus = 0.20 if input_skip else 0.0
    fitness = 100.0 - target_cost * 5.0 + recurrent_bonus + skip_bonus
    return FitnessResult(
        fitness=fitness,
        metrics={
            "proxy.target_cost": target_cost,
            "proxy.hidden_error": abs(hidden - 160.0),
            "proxy.decision_error": abs(decide_ticks - 10.0),
            "proxy.recurrent_bonus": recurrent_bonus,
            "proxy.hidden_layer_error": abs(hidden_layers - 2.0),
            "proxy.hidden_fanin_error": abs(hidden_fanin - 64.0),
            "proxy.input_skip_bonus": skip_bonus,
        },
    )


def _float_gene(genome: PolicyGenome, key: str) -> float:
    return float(genome.value(key))
