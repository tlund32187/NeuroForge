"""LIF/Surrogate spiking vision backbone composed from VisionBlockSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.core.torch_utils import require_torch
from neuroforge.vision.backbones.state import VisionState
from neuroforge.vision.blocks import SpikingConvBlock, SpikingPool, SpikingResBlock

if TYPE_CHECKING:
    from neuroforge.network.specs import VisionBackboneSpec

torch = require_torch()
nn = torch.nn

__all__ = ["LifConvNetV1"]


def _to_pair(value: object, *, default: int) -> int | tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, tuple | list):
        if len(value) != 2:
            msg = f"Expected int or 2-tuple/list, got {value!r}"
            raise ValueError(msg)
        return (int(value[0]), int(value[1]))
    msg = f"Expected int or 2-tuple/list, got {value!r}"
    raise ValueError(msg)


class LifConvNetV1(nn.Module):
    """Minimal spiking ConvNet backbone with temporal encoding support."""

    def __init__(self, spec: VisionBackboneSpec) -> None:
        super().__init__()
        if spec.type != "lif_convnet_v1":
            msg = f"LifConvNetV1 requires spec.type='lif_convnet_v1', got {spec.type!r}"
            raise ValueError(msg)

        self.type = spec.type
        self.input = spec.input
        self.time_steps = spec.time_steps
        self.encoding_mode = spec.encoding_mode
        self.output_dim = spec.output_dim

        blocks: list[Any] = []
        block_names: list[str] = []
        in_channels = int(spec.input.channels)

        for idx, block in enumerate(spec.blocks):
            params = dict(block.params)
            if block.type == "conv":
                out_channels = int(params["out_channels"])
                norm_raw = params.get("norm")
                norm = str(norm_raw) if norm_raw is not None else None
                layer = SpikingConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=_to_pair(params.get("kernel_size"), default=3),
                    stride=_to_pair(params.get("stride"), default=1),
                    padding=(
                        _to_pair(params.get("padding"), default=1)
                        if "padding" in params
                        else None
                    ),
                    bias=bool(params.get("bias", True)),
                    norm=norm,
                    norm_groups=int(params.get("norm_groups", 8)),
                    spike_threshold=float(params.get("spike_threshold", 0.0)),
                    spike_beta=float(params.get("spike_beta", 5.0)),
                )
                blocks.append(layer)
                block_names.append(f"conv_{idx}")
                in_channels = out_channels
                continue

            if block.type == "pool":
                kernel_size = _to_pair(params.get("kernel_size"), default=2)
                default_stride = (
                    int(kernel_size)
                    if isinstance(kernel_size, int)
                    else int(kernel_size[0])
                )
                layer = SpikingPool(
                    mode=str(params.get("mode", "max")),
                    kernel_size=kernel_size,
                    stride=_to_pair(params.get("stride"), default=default_stride),
                    padding=_to_pair(params.get("padding"), default=0),
                )
                blocks.append(layer)
                block_names.append(f"pool_{idx}")
                continue

            if block.type == "res":
                depth = int(params.get("depth", 1))
                out_channels = int(params.get("channels", in_channels))
                stride = _to_pair(params.get("stride"), default=1)
                norm_raw = params.get("norm", "batch")
                norm = str(norm_raw) if norm_raw is not None else None
                for d in range(depth):
                    this_stride: int | tuple[int, int] = stride if d == 0 else 1
                    layer = SpikingResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=_to_pair(params.get("kernel_size"), default=3),
                        stride=this_stride,
                        padding=(
                            _to_pair(params.get("padding"), default=1)
                            if "padding" in params
                            else None
                        ),
                        bias=bool(params.get("bias", False)),
                        norm=norm,
                        norm_groups=int(params.get("norm_groups", 8)),
                        spike_threshold=float(params.get("spike_threshold", 0.0)),
                        spike_beta=float(params.get("spike_beta", 5.0)),
                    )
                    blocks.append(layer)
                    block_names.append(f"res_{idx}_{d}")
                    in_channels = out_channels
                continue

            msg = f"Unsupported VisionBlockSpec.type: {block.type!r}"
            raise ValueError(msg)

        self.blocks = nn.ModuleList(blocks)
        self.block_names = tuple(block_names)
        self.head = nn.Linear(in_channels, self.output_dim)

    def _encode(self, x: Any) -> Any:
        """Encode static images into a ``[T, B, C, H, W]`` tensor stream."""
        x_prob = x.clamp(0.0, 1.0)
        t = int(self.time_steps)
        if self.encoding_mode == "constant":
            return x.unsqueeze(0).expand(t, -1, -1, -1, -1)
        if self.encoding_mode == "rate":
            thresholds = (torch.arange(t, device=x.device, dtype=x.dtype) + 0.5) / float(t)
            thresholds = thresholds.view(t, 1, 1, 1, 1)
            return (x_prob.unsqueeze(0) >= thresholds).to(x.dtype)
        if self.encoding_mode == "poisson":
            rand = torch.rand(
                (t, *x.shape),
                device=x.device,
                dtype=x_prob.dtype,
            )
            return (rand < x_prob.unsqueeze(0)).to(x.dtype)
        msg = f"Unsupported encoding_mode: {self.encoding_mode!r}"
        raise ValueError(msg)

    def _new_state(self, x: Any) -> VisionState:
        counts = {
            name: torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for name in self.block_names
        }
        neurons = {name: 0 for name in self.block_names}
        return VisionState(
            per_layer_spike_counts=counts,
            per_layer_neuron_count=neurons,
            time_steps=0,
            encoding_mode=self.encoding_mode,
        )

    def forward(
        self,
        x: Any,
        state: VisionState | None = None,
    ) -> tuple[Any, VisionState]:
        """Run T-step temporal encoding and block-stack inference.

        Returns
        -------
        tuple[Tensor, VisionState]
            ``features`` with shape ``[B, output_dim]`` and summary state.
        """
        encoded: Any
        runtime_time_steps: int
        if x.ndim == 4:
            if int(x.shape[1]) != int(self.input.channels):
                msg = (
                    "Input channel mismatch: "
                    f"expected {self.input.channels}, got {x.shape[1]}"
                )
                raise ValueError(msg)
            if (
                int(x.shape[2]) != int(self.input.height)
                or int(x.shape[3]) != int(self.input.width)
            ):
                msg = (
                    "Input spatial mismatch: expected "
                    f"({self.input.height}, {self.input.width}), got ({x.shape[2]}, {x.shape[3]})"
                )
                raise ValueError(msg)
            encoded = self._encode(x)
            runtime_time_steps = int(self.time_steps)
        elif x.ndim == 5:
            # Pre-binned event tensors are expected as [B, T, C, H, W].
            if int(x.shape[2]) != int(self.input.channels):
                msg = (
                    "Input channel mismatch: "
                    f"expected {self.input.channels}, got {x.shape[2]}"
                )
                raise ValueError(msg)
            if (
                int(x.shape[3]) != int(self.input.height)
                or int(x.shape[4]) != int(self.input.width)
            ):
                msg = (
                    "Input spatial mismatch: expected "
                    f"({self.input.height}, {self.input.width}), got ({x.shape[3]}, {x.shape[4]})"
                )
                raise ValueError(msg)
            runtime_time_steps = int(x.shape[1])
            if runtime_time_steps <= 0:
                msg = "Temporal dimension T must be > 0 for event tensors"
                raise ValueError(msg)
            encoded = x.permute(1, 0, 2, 3, 4).contiguous()
        else:
            msg = f"Expected input shape [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}"
            raise ValueError(msg)

        if state is None:
            state = self._new_state(encoded[0])
        else:
            batch_size = int(encoded.shape[1])
            for name in self.block_names:
                counts = state.per_layer_spike_counts.get(name)
                if (
                    counts is None
                    or counts.ndim != 1
                    or int(counts.shape[0]) != batch_size
                ):
                    state.per_layer_spike_counts[name] = torch.zeros(
                        batch_size,
                        dtype=encoded.dtype,
                        device=encoded.device,
                    )
                else:
                    state.per_layer_spike_counts[name] = counts.to(
                        device=encoded.device,
                        dtype=encoded.dtype,
                    )
                    state.per_layer_spike_counts[name].zero_()
                state.per_layer_neuron_count[name] = 0
            state.time_steps = 0
            state.encoding_mode = self.encoding_mode

        final_accum: Any = None
        for step in range(runtime_time_steps):
            y = encoded[step]
            for name, block in zip(self.block_names, self.blocks, strict=True):
                y = block(y)
                state.per_layer_neuron_count[name] = int(
                    y.shape[1] * y.shape[2] * y.shape[3]
                )
                state.per_layer_spike_counts[name] = (
                    state.per_layer_spike_counts[name]
                    + y.detach().sum(dim=(1, 2, 3))
                )
            final_accum = y if final_accum is None else final_accum + y
            state.time_steps += 1

        if final_accum is None:
            msg = "LifConvNetV1 has no blocks to execute"
            raise ValueError(msg)

        temporal_mean = final_accum / float(max(1, runtime_time_steps))
        pooled = temporal_mean.mean(dim=(2, 3))
        features = self.head(pooled)
        return features, state
