"""Online reward-modulated STDP for the live game loop.

Wires the existing :class:`RSTDPRule` into real-time play. Each plastic
projection keeps an eligibility trace; per engine tick the trace advances from
pre/post coincidences (no reward yet), and once per frame the (shaped) reward
converts the surviving trace into a weight change. The eligibility time constant
``tau_e`` bridges the gap between an action and the reward it earns several
frames later (three-factor learning: pre/post tag + delayed dopamine).

Dale-safe: weights are non-negative magnitudes clamped to ``[w_min=0, w_max]``,
so the synapse's ``|w|*sign`` polarity never flips. The reward is shaped into a
reward-prediction-error (scale -> subtract running baseline -> clip) so steady
reward causes no drift and one death cannot blow up the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuroforge.biology.plasticity.rules.rstdp import RSTDPParams, RSTDPRule
from neuroforge.contracts.biology.plasticity import LearningBatch

if TYPE_CHECKING:
    from neuroforge.simulation.engine.core import CoreEngine

__all__ = ["OnlineRSTDPConfig", "OnlineRSTDPTrainer", "RewardShaper"]


@dataclass(frozen=True, slots=True)
class OnlineRSTDPConfig:
    """Configuration for :class:`OnlineRSTDPTrainer`."""

    lr: float = 3e-4
    tau_e: float = 80e-3          # eligibility horizon (~ action->reward latency)
    a_plus: float = 1.0
    a_minus: float = 1.0
    w_min: float = 0.0            # Dale-safe: magnitudes stay non-negative
    w_max: float = 2.0
    reward_scale: float = 0.01
    reward_clip: float = 1.0
    baseline_beta: float = 0.01   # EMA rate for the reward baseline (0 disables)
    update_every_frames: int = 1
    consolidation_strength: float = 0.0
    dt: float = 1e-3
    plastic_projections: tuple[str, ...] = ("in_to_hidden", "hidden_to_motor")


class RewardShaper:
    """Turn raw reward into a clipped reward-prediction-error modulator."""

    def __init__(self, *, scale: float, clip: float, beta: float) -> None:
        self._scale = float(scale)
        self._clip = float(clip)
        self._beta = float(beta)
        self._baseline = 0.0

    @property
    def baseline(self) -> float:
        return self._baseline

    def reset(self) -> None:
        self._baseline = 0.0

    def shape(self, raw_reward: float) -> float:
        scaled = float(raw_reward) * self._scale
        advantage = scaled - self._baseline
        if self._beta > 0.0:
            self._baseline = (1.0 - self._beta) * self._baseline + self._beta * scaled
        return max(-self._clip, min(self._clip, advantage))


@dataclass
class _PlasticProjection:
    """Per-projection learning state owned by the trainer."""

    name: str
    proj: Any            # engine Projection (has .topology, .source, .target)
    pre_pop: Any         # source Population
    post_pop: Any        # target Population
    rule: RSTDPRule
    state: dict[str, Any]
    pending: Any         # accumulated dw [E] between weight writes
    loaded_weight_mask: Any | None = None
    consolidation_anchor: Any | None = None
    consolidation_mask: Any | None = None


class OnlineRSTDPTrainer:
    """Apply online reward-modulated STDP to a policy network's projections."""

    def __init__(
        self,
        engine: CoreEngine,
        config: OnlineRSTDPConfig | None = None,
        *,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        from neuroforge.kernel.torch_utils import require_torch

        self._torch = require_torch()
        self._cfg = config or OnlineRSTDPConfig()
        self._engine = engine
        self._frame = 0
        self._consolidation_strength = max(0.0, float(self._cfg.consolidation_strength))
        self._shaper = RewardShaper(
            scale=self._cfg.reward_scale,
            clip=self._cfg.reward_clip,
            beta=self._cfg.baseline_beta,
        )

        params = RSTDPParams(
            lr=self._cfg.lr,
            tau_e=self._cfg.tau_e,
            a_plus=self._cfg.a_plus,
            a_minus=self._cfg.a_minus,
            w_min=self._cfg.w_min,
            w_max=self._cfg.w_max,
        )
        projections = engine.projections
        populations = engine.populations
        self._plastic: list[_PlasticProjection] = []
        for name in self._cfg.plastic_projections:
            if name not in projections:
                msg = f"plastic projection {name!r} not found in engine"
                raise ValueError(msg)
            proj = projections[name]
            rule = RSTDPRule(params)
            n_edges = int(proj.topology.pre_idx.numel())
            state = rule.init_state(n_edges, device, dtype)
            self._plastic.append(
                _PlasticProjection(
                    name=name,
                    proj=proj,
                    pre_pop=populations[proj.source],
                    post_pop=populations[proj.target],
                    rule=rule,
                    state=state,
                    pending=self._torch.zeros_like(state["eligibility"]),
                )
            )

    @property
    def plastic_names(self) -> tuple[str, ...]:
        return tuple(pp.name for pp in self._plastic)

    @property
    def reward_baseline(self) -> float:
        return self._shaper.baseline

    def begin_episode(self) -> None:
        """Zero eligibility/pending and reset the reward baseline."""
        for pp in self._plastic:
            pp.state["eligibility"].zero_()
            pp.pending.zero_()
        self._shaper.reset()
        self._frame = 0

    def accumulate(self) -> None:
        """Advance every eligibility trace one tick from current pre/post spikes.

        Designed to be the engine's per-tick ``on_tick`` callback. Reward is
        zero here â€” only the synaptic tag is laid down.
        """
        torch = self._torch
        with torch.no_grad():
            for pp in self._plastic:
                pre_full = pp.pre_pop.state.get("last_spikes")
                post_full = pp.post_pop.state.get("last_spikes")
                if pre_full is None or post_full is None:
                    continue
                topo = pp.proj.topology
                batch = LearningBatch(
                    pre_spikes=pre_full[topo.pre_idx],
                    post_spikes=post_full[topo.post_idx],
                    weights=topo.weights,
                    eligibility=pp.state["eligibility"],
                    reward=None,
                )
                pp.rule.step(pp.state, batch, self._cfg.dt)  # advances state["eligibility"]

    def learn(self, raw_reward: float) -> dict[str, float]:
        """Convert the surviving eligibility into a weight change for one frame."""
        shaped = self._shaper.shape(raw_reward)
        self._frame += 1
        dw_norm = 0.0
        consolidation_norm = 0.0
        torch = self._torch
        with torch.no_grad():
            for pp in self._plastic:
                dw = pp.rule.params.lr * shaped * pp.state["eligibility"]
                pp.pending = pp.pending + dw
                dw_norm += float(dw.abs().sum().item())

            if self._frame % max(1, self._cfg.update_every_frames) == 0:
                for pp in self._plastic:
                    topo = pp.proj.topology
                    topo.weights.copy_(pp.rule.apply_dw(topo.weights, pp.pending))
                    consolidation_norm += self._apply_consolidation(pp)
                    pp.pending.zero_()

        return {
            "reward_raw": float(raw_reward),
            "reward_shaped": float(shaped),
            "reward_baseline": float(self._shaper.baseline),
            "dw_norm": dw_norm,
            "consolidation_norm": consolidation_norm,
        }

    #

    def _apply_consolidation(self, pp: _PlasticProjection) -> float:
        """Blend anchored weights back toward their checkpoint values."""
        anchor = pp.consolidation_anchor
        mask = pp.consolidation_mask
        if anchor is None or mask is None:
            return 0.0
        if self._consolidation_strength <= 0.0:
            return 0.0
        weights = pp.proj.topology.weights
        delta = (anchor - weights) * mask * self._consolidation_strength
        if bool(mask.any()):
            weights.add_(delta).clamp_(self._cfg.w_min, self._cfg.w_max)
        return float(delta.abs().sum().item())

    def anchor_current_weights(
        self,
        *,
        strength: float | None = None,
        loaded_only: bool = True,
    ) -> None:
        """Anchor current synapses so later training resists catastrophic drift.

        ``loaded_only`` anchors only weights restored from a checkpoint. That is
        important for partial warm-starts: old knowledge is protected, while
        newly added capacity remains free to adapt.
        """
        next_strength = self._cfg.consolidation_strength if strength is None else float(strength)
        self._consolidation_strength = max(0.0, float(next_strength))
        if self._consolidation_strength <= 0.0:
            return
        for pp in self._plastic:
            weights = pp.proj.topology.weights
            mask = pp.loaded_weight_mask if loaded_only else self._torch.ones_like(weights)
            if mask is None:
                continue
            pp.consolidation_anchor = weights.detach().clone()
            pp.consolidation_mask = mask.to(weights.device, weights.dtype).clone()

    def weights_snapshot(self) -> dict[str, Any]:
        return {pp.name: pp.proj.topology.weights.detach().cpu().clone() for pp in self._plastic}

    def eligibility_snapshot(self) -> dict[str, Any]:
        return {pp.name: pp.state["eligibility"].detach().cpu().clone() for pp in self._plastic}

    def load_weights(
        self,
        snapshot: dict[str, Any],
        *,
        allow_partial: bool = False,
    ) -> dict[str, dict[str, int | bool | str]]:
        summary: dict[str, dict[str, int | bool | str]] = {}
        for pp in self._plastic:
            target = pp.proj.topology.weights
            if pp.name not in snapshot:
                summary[pp.name] = _missing_load_summary(target)
                continue
            source_numel, copied, partial = _load_tensor(
                target,
                snapshot[pp.name],
                allow_partial=allow_partial,
            )
            loaded_mask = self._torch.zeros_like(target, dtype=self._torch.bool)
            _mark_overlap_1d(loaded_mask, copied)
            pp.loaded_weight_mask = loaded_mask
            summary[pp.name] = _loaded_summary(
                target,
                source_numel=source_numel,
                copied_numel=copied,
                partial=partial,
            )
        return summary

    def load_eligibility(
        self,
        snapshot: dict[str, Any],
        *,
        allow_partial: bool = False,
    ) -> dict[str, dict[str, int | bool | str]]:
        summary: dict[str, dict[str, int | bool | str]] = {}
        for pp in self._plastic:
            target = pp.state["eligibility"]
            if pp.name not in snapshot:
                summary[pp.name] = _missing_load_summary(target)
                continue
            source_numel, copied, partial = _load_tensor(
                target,
                snapshot[pp.name],
                allow_partial=allow_partial,
            )
            summary[pp.name] = _loaded_summary(
                target,
                source_numel=source_numel,
                copied_numel=copied,
                partial=partial,
            )
        return summary


def _load_tensor(
    target: Any,
    source_raw: Any,
    *,
    allow_partial: bool,
) -> tuple[int, int, bool]:
    """Load one checkpoint tensor into *target* and report copy coverage."""
    source = source_raw.to(target.device, target.dtype).reshape(-1)
    source_numel = int(source.numel())
    target_numel = int(target.numel())
    if allow_partial and tuple(source.shape) != tuple(target.shape):
        copied = _copy_overlap_1d(target, source)
        return source_numel, copied, True
    target.copy_(source.reshape_as(target))
    return source_numel, target_numel, False


def _missing_load_summary(target: Any) -> dict[str, int | bool | str]:
    return {
        "loaded": False,
        "reason": "missing",
        "source_numel": 0,
        "target_numel": int(target.numel()),
        "copied_numel": 0,
        "partial": False,
    }


def _loaded_summary(
    target: Any,
    *,
    source_numel: int,
    copied_numel: int,
    partial: bool,
) -> dict[str, int | bool | str]:
    return {
        "loaded": copied_numel > 0,
        "reason": "",
        "source_numel": int(source_numel),
        "target_numel": int(target.numel()),
        "copied_numel": int(copied_numel),
        "partial": bool(partial),
    }


def _copy_overlap_1d(target: Any, source: Any) -> int:
    """Copy the overlapping prefix from *source* into a flattened *target* tensor."""
    flat_target = target.reshape(-1)
    n = min(int(flat_target.numel()), int(source.numel()))
    if n > 0:
        flat_target[:n].copy_(source[:n])
    return n


def _mark_overlap_1d(mask: Any, n: int) -> None:
    """Mark the first *n* flattened entries as loaded from a checkpoint."""
    flat_mask = mask.reshape(-1)
    if n > 0:
        flat_mask[:n].fill_(True)
