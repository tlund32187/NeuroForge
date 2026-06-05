"""Network specification DTOs for declarative network construction.

These frozen dataclasses describe the structure of a spiking neural
network (populations, projections, topology) without performing any
tensor allocation or I/O. They are consumed by :class:`NetworkFactory`
to produce a fully initialised :class:`CoreEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

__all__ = [
    "GateNetworkSpec",
    "PopulationSpec",
    "ProjectionSpec",
    "NetworkSpec",
    "VisionBackboneSpec",
    "VisionBlockSpec",
    "VisionInputSpec",
]

_VISION_ENCODING_MODES: frozenset[str] = frozenset({"rate", "poisson", "constant"})
_VISION_BLOCK_TYPES: frozenset[str] = frozenset({"conv", "res", "pool"})


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_positive_int(value: object, *, field_name: str) -> None:
    if not _is_positive_int(value):
        msg = f"{field_name} must be a positive integer; got {value!r}"
        raise ValueError(msg)


def _validate_kernel_size(value: object, *, field_name: str) -> None:
    if _is_positive_int(value):
        return
    if isinstance(value, tuple | list):
        val = cast("Any", value)
        if len(val) != 2:
            msg = f"{field_name} must be an int or a 2-tuple; got {value!r}"
            raise ValueError(msg)
        if all(_is_positive_int(v) for v in val):
            return
    msg = f"{field_name} must be an int or a 2-tuple of positive integers; got {value!r}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class VisionInputSpec:
    """Input tensor shape for a vision backbone."""

    channels: int
    height: int
    width: int

    def __post_init__(self) -> None:
        _validate_positive_int(self.channels, field_name="VisionInputSpec.channels")
        _validate_positive_int(self.height, field_name="VisionInputSpec.height")
        _validate_positive_int(self.width, field_name="VisionInputSpec.width")


@dataclass(frozen=True, slots=True)
class VisionBlockSpec:
    """One block in a vision backbone pipeline."""

    type: str
    params: dict[str, Any] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        block_type = self.type.strip().lower()
        if block_type not in _VISION_BLOCK_TYPES:
            msg = (
                "VisionBlockSpec.type must be one of "
                f"{sorted(_VISION_BLOCK_TYPES)}; got {self.type!r}"
            )
            raise ValueError(msg)
        if not isinstance(self.params, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = "VisionBlockSpec.params must be a dict[str, Any]"
            raise ValueError(msg)
        object.__setattr__(self, "type", block_type)

        params = dict(self.params)
        object.__setattr__(self, "params", params)
        if block_type == "conv":
            if "out_channels" not in params:
                msg = "VisionBlockSpec(type='conv') missing required param: 'out_channels'"
                raise ValueError(msg)
            if "kernel_size" not in params:
                msg = "VisionBlockSpec(type='conv') missing required param: 'kernel_size'"
                raise ValueError(msg)
            _validate_positive_int(
                params["out_channels"],
                field_name="VisionBlockSpec.params['out_channels']",
            )
            _validate_kernel_size(
                params["kernel_size"],
                field_name="VisionBlockSpec.params['kernel_size']",
            )
            if "stride" in params:
                _validate_kernel_size(
                    params["stride"],
                    field_name="VisionBlockSpec.params['stride']",
                )
        elif block_type == "res":
            if "channels" in params:
                _validate_positive_int(
                    params["channels"],
                    field_name="VisionBlockSpec.params['channels']",
                )
            if "depth" in params:
                _validate_positive_int(
                    params["depth"],
                    field_name="VisionBlockSpec.params['depth']",
                )
        elif block_type == "pool":
            if "kernel_size" not in params:
                msg = "VisionBlockSpec(type='pool') missing required param: 'kernel_size'"
                raise ValueError(msg)
            _validate_kernel_size(
                params["kernel_size"],
                field_name="VisionBlockSpec.params['kernel_size']",
            )
            if "stride" in params:
                _validate_kernel_size(
                    params["stride"],
                    field_name="VisionBlockSpec.params['stride']",
                )
            mode = str(params.get("mode", "max")).lower()
            if mode not in {"max", "avg"}:
                msg = (
                    "VisionBlockSpec(type='pool').params['mode'] must be "
                    f"'max' or 'avg'; got {mode!r}"
                )
                raise ValueError(msg)
            params["mode"] = mode
            object.__setattr__(self, "params", params)


@dataclass(frozen=True, slots=True)
class VisionBackboneSpec:
    """Config DTO for selecting and shaping a vision backbone."""

    type: str
    input: VisionInputSpec
    time_steps: int
    encoding_mode: str = "rate"
    blocks: list[VisionBlockSpec] = field(default_factory=lambda: [])
    output_dim: int = 1

    @property
    def T(self) -> int:
        """Alias for ``time_steps`` used in config docs."""
        return self.time_steps

    def __post_init__(self) -> None:
        backbone_type = self.type.strip()
        if not backbone_type:
            msg = "VisionBackboneSpec.type must be a non-empty string"
            raise ValueError(msg)
        object.__setattr__(self, "type", backbone_type)

        if not isinstance(self.input, VisionInputSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
            if isinstance(self.input, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                raw = cast("dict[object, Any]", self.input)
                try:
                    parsed_input = VisionInputSpec(
                        channels=raw["channels"],
                        height=raw["height"],
                        width=raw["width"],
                    )
                except KeyError as exc:
                    msg = "VisionBackboneSpec.input must include channels, height, width"
                    raise ValueError(msg) from exc
                object.__setattr__(self, "input", parsed_input)
            else:
                msg = (
                    "VisionBackboneSpec.input must be a VisionInputSpec "
                    "or dict with channels/height/width"
                )
                raise ValueError(msg)

        _validate_positive_int(self.time_steps, field_name="VisionBackboneSpec.time_steps")
        _validate_positive_int(self.output_dim, field_name="VisionBackboneSpec.output_dim")

        encoding_mode = self.encoding_mode.strip().lower()
        if encoding_mode not in _VISION_ENCODING_MODES:
            msg = (
                "VisionBackboneSpec.encoding_mode must be one of "
                f"{sorted(_VISION_ENCODING_MODES)}; got {self.encoding_mode!r}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "encoding_mode", encoding_mode)

        if not isinstance(self.blocks, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = "VisionBackboneSpec.blocks must be a list[VisionBlockSpec]"
            raise ValueError(msg)
        if not self.blocks:
            msg = "VisionBackboneSpec.blocks must contain at least one block"
            raise ValueError(msg)

        parsed_blocks: list[VisionBlockSpec] = []
        for idx, block in enumerate(self.blocks):
            if isinstance(block, VisionBlockSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
                parsed_blocks.append(block)
                continue
            if isinstance(block, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                raw_block = cast("dict[object, Any]", block)
                block_type = raw_block.get("type")
                if block_type is None:
                    msg = f"VisionBackboneSpec.blocks[{idx}] missing required key: 'type'"
                    raise ValueError(msg)

                raw_params = raw_block.get("params", {})
                if raw_params is None:
                    raw_params = {}
                if not isinstance(raw_params, dict):
                    msg = (
                        f"VisionBackboneSpec.blocks[{idx}].params must be a dict[str, Any]"
                    )
                    raise ValueError(msg)
                params = {str(k): v for k, v in cast("dict[object, Any]", raw_params).items()}
                parsed_blocks.append(VisionBlockSpec(type=str(block_type), params=params))
                continue

            msg = (
                f"VisionBackboneSpec.blocks[{idx}] must be VisionBlockSpec "
                "or dict[type, params]"
            )
            raise ValueError(msg)

        object.__setattr__(self, "blocks", parsed_blocks)


@dataclass(frozen=True, slots=True)
class PopulationSpec:
    """Specification for a neuron population.

    Attributes
    ----------
    name:
        Unique population identifier (e.g. ``"input"``, ``"hidden"``).
    n:
        Number of neurons.
    neuron_model:
        Registry key for the neuron model (e.g. ``"lif"``).
    neuron_params:
        Keyword arguments forwarded to the neuron model constructor.
    """

    name: str
    n: int
    neuron_model: str
    neuron_params: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Specification for a synaptic projection between two populations.

    Attributes
    ----------
    name:
        Unique projection identifier.
    source:
        Name of the pre-synaptic population.
    target:
        Name of the post-synaptic population.
    synapse_model:
        Registry key for the synapse model (e.g. ``"static"``).
    synapse_params:
        Keyword arguments forwarded to the synapse model constructor.
    topology:
        Wiring and initialisation configuration.

        ``type`` can be:

        - ``"dense"``: full all-to-all matrix (existing behavior)
        - ``"sparse_random"``: Bernoulli sampling with ``p_connect``
        - ``"sparse_fanout"``: fixed ``fanout`` edges per pre-neuron
        - ``"sparse_fanin"``: fixed ``fanin`` edges per post-neuron
        - ``"block_sparse"``: sampled dense blocks with
          ``block_pre``, ``block_post``, ``p_block``

        Common keys for sparse topologies:

        - ``init``: ``"uniform"`` (default) or ``"zeros"``
        - ``low`` / ``high``: bounds for uniform init (default +/-0.3)
        - ``bias``: ``True`` to allocate per-target bias
        - ``delays``: optional dict
          ``{"mode": "zeros" | "uniform_int", "max_delay": int}``
        - ``sort``: reorder sparse edges for locality (default ``True``)
        - ``seed``: optional per-projection seed override
          (defaults to engine seed)
    """

    name: str
    source: str
    target: str
    synapse_model: str
    synapse_params: dict[str, Any] = field(default_factory=lambda: {})
    topology: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    """Full network specification (populations + projections).

    Attributes
    ----------
    populations:
        Ordered list of population specifications.
    projections:
        Ordered list of projection specifications.
    metadata:
        Arbitrary extra data (task name, version, notes, etc.).
    """

    populations: list[PopulationSpec]
    projections: list[ProjectionSpec]
    metadata: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class GateNetworkSpec:
    """High-level specification for a logic-gate spiking network.

    Describes the network shape and initialisation policy without
    allocating any tensors. Consumed by :func:`build_gate_network`.

    Attributes
    ----------
    input_size:
        Number of input neurons (e.g. 2 for binary gates).
    hidden_size:
        Number of hidden neurons. Set to 0 for the no-hidden
        (input -> output) case.
    output_size:
        Number of output neurons (1 for single-gate tasks).
    n_inhibitory_hidden:
        How many of the hidden neurons are inhibitory
        (Dale's Law). Ignored when ``hidden_size == 0``.
    seed:
        Random seed for reproducible weight initialisation.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    dtype:
        Torch dtype string (``"float32"`` or ``"float64"``).
    init_scale:
        Symmetric uniform init range ``[-init_scale, +init_scale]``.
    p_connect:
        Connection probability per possible edge. ``1.0`` gives
        fully-connected (dense) wiring; values < 1 produce a
        randomly-sampled sparse topology (seeded).
    neuron_model:
        Registry key for the neuron model used in all populations
        (e.g. ``"lif_surr"``).
    synapse_model:
        Registry key for the synapse model used in all projections
        (e.g. ``"static_dales"``).
    """

    input_size: int = 2
    hidden_size: int = 6
    output_size: int = 1
    n_inhibitory_hidden: int = 2
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    init_scale: float = 0.3
    p_connect: float = 1.0
    neuron_model: str = "lif_surr"
    synapse_model: str = "static_dales"
