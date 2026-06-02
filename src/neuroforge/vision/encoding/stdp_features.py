# pyright: basic
"""STDP competitive feature maps (Layer A1): unsupervised local feature discovery.

Builds on A0's ON/OFF contrast ([[retina]]). A bank of feature neurons each owns
a small receptive field over the contrast map. Shown a patch, they compete via
lateral inhibition (winner-take-all) and only the winner learns, by an
STDP-derived rule — so a set of *differentiated* local feature detectors
(oriented edges, corners, sprite-parts, block/pipe textures) emerges with NO
labels and NO prepopulated shapes.

Spike-timing grounding (rank-order STDP, Masquelier & Thorpe 2007): stronger
contrast => earlier spike; a feature neuron's membrane sums the weights of the
inputs that have spiked so far, so the (norm-corrected) matched-filter response
``weights·patch`` predicts which neuron reaches threshold first. That
first-to-fire neuron wins, inhibits the rest, and undergoes STDP — synapses whose
input spiked *before* it fired (high contrast) are potentiated, the rest
depressed (``a_minus < a_plus``). A win-frequency "conscience" (homeostasis)
keeps every feature in use, so there are no dead or runaway-dominant units.

The learner is patch-based and self-contained (mirrors :class:`RetinaEncoder`);
``train_on_contrast``/``feature_map`` chain it to A0's output. Wiring the pooled
feature code in as the policy drive is A4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuroforge.core.torch_utils import require_torch, resolve_device_dtype

__all__ = ["STDPFeatureConfig", "STDPFeatureLayer"]


@dataclass(frozen=True, slots=True)
class STDPFeatureConfig:
    """Configuration for :class:`STDPFeatureLayer`."""

    n_features: int = 24
    patch: int = 5               # receptive-field side on the contrast grid
    in_channels: int = 2         # ON/OFF from the retina
    lr: float = 0.02
    a_plus: float = 1.0          # potentiation of inputs that spiked before the win
    a_minus: float = 0.6         # depression of the rest (asymmetric: < a_plus)
    w_min: float = 0.0
    w_max: float = 1.0
    conscience: float = 0.5      # homeostasis strength (0 disables); keeps features in use
    stride: int = 2             # patch sampling stride
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.n_features < 2:
            msg = "STDPFeatureConfig.n_features must be >= 2"
            raise ValueError(msg)
        if self.patch < 1:
            msg = "STDPFeatureConfig.patch must be >= 1"
            raise ValueError(msg)
        if self.in_channels < 1:
            msg = "STDPFeatureConfig.in_channels must be >= 1"
            raise ValueError(msg)
        if self.stride < 1:
            msg = "STDPFeatureConfig.stride must be >= 1"
            raise ValueError(msg)
        if not self.w_min < self.w_max:
            msg = "STDPFeatureConfig requires w_min < w_max"
            raise ValueError(msg)


class STDPFeatureLayer:
    """A bank of competing feature detectors trained by rank-order STDP."""

    def __init__(self, config: STDPFeatureConfig | None = None) -> None:
        self._cfg = config or STDPFeatureConfig()
        self._torch = require_torch()
        self._dev, self._dtype = resolve_device_dtype(self._cfg.device, self._cfg.dtype)
        self._dim = self._cfg.in_channels * self._cfg.patch * self._cfg.patch
        gen = self._torch.Generator(device="cpu")
        gen.manual_seed(int(self._cfg.seed))
        w = self._torch.rand(
            self._cfg.n_features, self._dim, generator=gen, dtype=self._dtype,
        )
        self._w = (w * self._cfg.w_max).to(self._dev)
        self._win_count = self._torch.zeros(
            self._cfg.n_features, dtype=self._dtype, device=self._dev,
        )
        self._total_wins = 0.0

    @property
    def n_features(self) -> int:
        return self._cfg.n_features

    @property
    def weights(self) -> Any:
        """Feature weights reshaped to ``[n_features, in_channels, patch, patch]``."""
        return self._w.reshape(
            self._cfg.n_features, self._cfg.in_channels, self._cfg.patch, self._cfg.patch,
        ).clone()

    def reset_homeostasis(self) -> None:
        self._win_count.zero_()
        self._total_wins = 0.0

    def state_dict(self) -> dict[str, Any]:
        """Serialisable learned state (for checkpointing alongside the policy)."""
        return {
            "w": self._w.detach().cpu().clone(),
            "win_count": self._win_count.detach().cpu().clone(),
            "total_wins": float(self._total_wins),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore learned state produced by :meth:`state_dict`."""
        if "w" in state:
            self._w = state["w"].to(self._dev, self._dtype)
        if "win_count" in state:
            self._win_count = state["win_count"].to(self._dev, self._dtype)
        if "total_wins" in state:
            self._total_wins = float(state["total_wins"])

    # ── core: response, competition, learning ─────────────────────────

    def responses(self, patches: Any) -> Any:
        """Norm-corrected matched-filter response ``[N, n_features]`` (no conscience)."""
        norms = self._w.norm(dim=1) + 1e-6
        return (patches @ self._w.t()) / norms

    def winners(self, patches: Any) -> Any:
        """Winning feature index per patch ``[N]`` (after the conscience bias)."""
        return self._biased_responses(patches).argmax(dim=1)

    def partial_fit(self, patches: Any) -> Any:
        """One batch of competitive STDP. Returns the winners ``[N]``."""
        torch = self._torch
        cfg = self._cfg
        if patches.ndim != 2 or patches.shape[1] != self._dim:
            msg = f"patches must be [N, {self._dim}]"
            raise ValueError(msg)

        winners = self._biased_responses(patches).argmax(dim=1)
        onehot = torch.nn.functional.one_hot(winners, cfg.n_features).to(self._dtype)
        counts = onehot.sum(dim=0)                                   # [F]
        # Mean presynaptic activity (spiked-before-the-win) among each feature's wins.
        active = (patches > 0).to(self._dtype)                       # [N, D]
        active_mean = (onehot.t() @ active) / counts.clamp_min(1.0).unsqueeze(1)  # [F, D]

        won = counts > 0
        pot = cfg.lr * cfg.a_plus * active_mean * (cfg.w_max - self._w)
        dep = cfg.lr * cfg.a_minus * (1.0 - active_mean) * (self._w - cfg.w_min)
        delta = (pot - dep) * won.unsqueeze(1).to(self._dtype)
        self._w = (self._w + delta).clamp_(cfg.w_min, cfg.w_max)

        self._win_count += counts
        self._total_wins += float(counts.sum().item())
        return winners

    # ── frame-level convenience (chains A0 -> A1) ──────────────────────

    def train_on_contrast(self, contrast_chw: Any) -> Any:
        """Extract patches from an A0 contrast map ``[C, H, W]`` and fit one batch."""
        return self.partial_fit(self._extract_patches(contrast_chw))

    def feature_map(self, contrast_chw: Any) -> Any:
        """Per-feature responses over space: ``[n_features, H', W']``."""
        patches, out_h, out_w = self._extract_patches(contrast_chw, with_grid=True)
        resp = self.responses(patches)                               # [L, F]
        return resp.t().reshape(self._cfg.n_features, out_h, out_w)

    def encode(self, contrast_chw: Any) -> Any:
        """Translation-invariant feature vector: max-pool over space ``[n_features]``."""
        fmap = self.feature_map(contrast_chw)
        return fmap.reshape(self._cfg.n_features, -1).amax(dim=1)

    # ── internals ─────────────────────────────────────────────────────

    def _biased_responses(self, patches: Any) -> Any:
        resp = self.responses(patches)
        if self._cfg.conscience <= 0.0 or self._total_wins <= 0.0:
            return resp
        # "Conscience": penalise features that win more than their fair share, so
        # under-used features get a chance — homeostasis against dead/dominant units.
        freq = self._win_count / self._total_wins
        bias = self._cfg.conscience * (freq - 1.0 / self._cfg.n_features)
        return resp - bias.unsqueeze(0)

    def _extract_patches(
        self, contrast_chw: Any, *, with_grid: bool = False,
    ) -> Any:
        torch = self._torch
        cfg = self._cfg
        if contrast_chw.ndim != 3 or contrast_chw.shape[0] != cfg.in_channels:
            msg = f"contrast map must be [{cfg.in_channels}, H, W]"
            raise ValueError(msg)
        cols = torch.nn.functional.unfold(
            contrast_chw.unsqueeze(0), kernel_size=cfg.patch, stride=cfg.stride,
        )  # [1, C*patch*patch, L]
        patches = cols[0].t().contiguous()  # [L, D]
        if not with_grid:
            return patches
        _, height, width = contrast_chw.shape
        out_h = (height - cfg.patch) // cfg.stride + 1
        out_w = (width - cfg.patch) // cfg.stride + 1
        return patches, out_h, out_w
