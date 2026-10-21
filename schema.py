"""Backend-neutral facts. Tensor values and live modules never enter reports."""


from dataclasses import asdict, dataclass


from . import constants
@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class ModuleInfo:
    path: str
    class_name: str
    properties: dict[str, str | int | float | bool]
    tensors: tuple[TensorInfo, ...]
    alias_of: str | None = None

    @property
    def property_sources(self) -> dict[str, str]:
        return {
            key: constants.FIELD_QUANTIZATION_CONFIG if key == constants.FIELD_CONFIGURED_BITS else constants.SOURCE_MODULE_ATTRIBUTE
            for key in self.properties
        }


@dataclass(frozen=True)
class LayerInfo:
    index: int
    path: str
    attention_type: str
    hidden_size: int
    head_dim: int
    query_heads: int
    kv_heads: int
    intermediate_size: int
    kv_source: int
    shared_kv: bool
    sliding_window: int | None
    activation: str
    rope_theta: float
    rotary_fraction: float
    norm_epsilon: float
    attention_scale: float

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.attention_type, self.hidden_size, self.head_dim,
            self.query_heads, self.kv_heads, self.intermediate_size,
            self.shared_kv, self.sliding_window, self.activation,
            self.rope_theta, self.rotary_fraction, self.norm_epsilon, self.attention_scale,
        )


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    kind: str
    module_path: str | None = None


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str = constants.EDGE_DATA_FLOW


@dataclass(frozen=True)
class GraphView:
    id: str
    title: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    member_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ArchitectureGraph:
    overview: GraphView
    decoders: tuple[GraphView, ...]
    hierarchy: GraphView
    modules: tuple[ModuleInfo, ...]
    layers: tuple[LayerInfo, ...]
    evidence: str = constants.EVIDENCE_VERIFIED_ARCHITECTURE

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiagramArtifact:
    name: str
    content: str
    media_type: str
