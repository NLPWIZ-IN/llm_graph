"""Verified text path for the Gemma 4 E2B mobile checkpoint in Transformers 5.19."""


import re
from importlib.metadata import version
from typing import NoReturn

from torch import nn
from transformers import PreTrainedModel
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from transformers.integrations.gemma_quant import QuantizedEmbedding, QuantizedLinear
from transformers.models.gemma4 import modeling_gemma4 as g
from transformers.utils.quantization_config import GemmaQuantizationConfig

from . import constants
from .config import DiagramConfig
from .errors import UnsupportedArchitectureError
from .inspection import ModuleInspector
from .schema import ArchitectureGraph, Edge, GraphView, LayerInfo, Node


MODEL_ID = constants.GEMMA_MODEL_ID
PREFIX = constants.GEMMA_TEXT_PREFIX


class _Graph:
    def __init__(self, id_: str, title: str) -> None:
        self.id = id_
        self.title = title
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def add(self, id_: str, label: str, kind: str, path: str | None = None) -> str:
        self.nodes.append(Node(id_, label, kind, path))
        return id_

    def link(self, *ids: str, kind: str = constants.EDGE_DATA_FLOW) -> None:
        self.edges.extend(Edge(a, b, kind) for a, b in zip(ids, ids[1:]))

    def finish(self, members: tuple[int, ...] = ()) -> GraphView:
        return GraphView(self.id, self.title, tuple(self.nodes), tuple(self.edges), members)


def _layout(path: str) -> NoReturn:
    raise UnsupportedArchitectureError(
        constants.ERROR_ARCHITECTURE_LAYOUT, constants.STAGE_INSPECTION, adapter=constants.GEMMA_ADAPTER, component_path=path
    )


def _quant_dict(model: nn.Module) -> dict:
    if not isinstance(model, PreTrainedModel):
        _layout(constants.PATH_CONFIG)
    config = vars(model.config).get(constants.FIELD_QUANTIZATION_CONFIG)
    if isinstance(config, dict):
        return config
    if isinstance(config, GemmaQuantizationConfig):
        return config.to_dict()
    _layout(constants.PATH_QUANTIZATION_CONFIG)


def _module(model: nn.Module, path: str) -> nn.Module:
    try:
        return model.get_submodule(path)
    except AttributeError:
        _layout(path)


def configured_bits(model: nn.Module, path: str) -> int | None:
    """Mirror the installed quantizer's combined-regex search precedence."""
    from transformers.quantizers.quantizers_utils import should_convert_module

    config = _quant_dict(model)
    if not should_convert_module(path, config.get(constants.FIELD_MODULES_TO_NOT_CONVERT)):
        return None
    module = _module(model, path)
    if isinstance(module, (nn.Embedding, QuantizedEmbedding)):
        if not config.get(constants.FIELD_QUANTIZE_EMBEDDINGS):
            return None
    elif not isinstance(module, nn.Linear):
        return None
    overrides = config.get(constants.FIELD_MODULE_QUANT_CONFIGS) or {}
    bits = config[constants.FIELD_NUM_BITS]
    if overrides:
        matcher = re.compile(constants.REGEX_ALTERNATION.join(
            constants.TEMPLATE_QUANT_REGEX_GROUP.format(i, pattern) for i, pattern in enumerate(overrides)
        ))
        match = matcher.search(path)
        if match:
            values = list(overrides.values())
            for i in range(len(values)):
                if match.group(constants.TEMPLATE_QUANT_REGEX_GROUP_NAME.format(i)) is not None:
                    bits = values[i].get(constants.FIELD_NUM_BITS, bits)
                    break
    return int(bits)


class Gemma4Adapter:
    name = constants.GEMMA_ADAPTER

    def matches(self, model: nn.Module) -> bool:
        return type(model) is g.Gemma4ForConditionalGeneration

    def inspect(
        self, model: nn.Module, inspector: ModuleInspector, config: DiagramConfig
    ) -> ArchitectureGraph:
        installed_version = version(constants.TRANSFORMERS_DISTRIBUTION)
        if installed_version != constants.VERIFIED_TRANSFORMERS_VERSION:
            raise UnsupportedArchitectureError(
                constants.ERROR_UNSUPPORTED_TRANSFORMERS_VERSION, constants.STAGE_RESOLUTION, adapter=self.name,
                installed_version=installed_version,
            )
        if not isinstance(model, g.Gemma4ForConditionalGeneration):
            _layout(constants.PATH_MODEL)
        if not isinstance(model.config, Gemma4Config):
            _layout(constants.PATH_CONFIG)
        if type(model.model) is not g.Gemma4Model or constants.METHOD_FORWARD in vars(model.model):
            _layout(constants.PATH_MODEL)
        c = model.config.text_config
        if not isinstance(c, Gemma4TextConfig):
            _layout(constants.PATH_TEXT_CONFIG)
        expected = {
            constants.FIELD_HIDDEN_SIZE: 1536, constants.FIELD_NUM_HIDDEN_LAYERS: 35,
            constants.FIELD_INTERMEDIATE_SIZE: 6144, constants.FIELD_NUM_ATTENTION_HEADS: 8,
            constants.FIELD_HIDDEN_SIZE_PER_LAYER_INPUT: 256, constants.FIELD_NUM_KV_SHARED_LAYERS: 20,
            constants.FIELD_ENABLE_MOE_BLOCK: False, constants.FIELD_USE_DOUBLE_WIDE_MLP: True,
            constants.FIELD_ATTENTION_K_EQ_V: False, constants.FIELD_VOCAB_SIZE: 262144,
            constants.FIELD_VOCAB_SIZE_PER_LAYER_INPUT: 262144, constants.FIELD_SLIDING_WINDOW: 512,
            constants.FIELD_HIDDEN_ACTIVATION: constants.GEMMA_ACTIVATION,
        }
        if any(getattr(c, key, None) != value for key, value in expected.items()):
            raise UnsupportedArchitectureError(
                constants.ERROR_UNSUPPORTED_VARIANT, constants.STAGE_RESOLUTION, adapter=self.name,
                variant=constants.TEXT_CONFIGURATION_OUTSIDE_GEMMA_E2B_MOBILE_PROFILE,
            )
        expected_types = [
            constants.ATTENTION_FULL if i % 5 == 4 else constants.ATTENTION_SLIDING for i in range(35)
        ]
        if c.layer_types != expected_types or c.use_bidirectional_attention == constants.BIDIRECTIONAL_ALL:
            _layout(constants.PATH_LAYER_TYPES)
        quant = _quant_dict(model)
        if quant.get(constants.FIELD_QUANT_METHOD) != constants.GEMMA_QUANT_METHOD or quant.get(constants.FIELD_NUM_BITS) != 4:
            _layout(constants.PATH_QUANTIZATION_CONFIG)
        decoder = _module(model, PREFIX)
        if not isinstance(decoder, g.Gemma4TextModel) or len(decoder.layers) != 35:
            _layout(PREFIX)
        for path, dimensions in {
            PREFIX + constants.SUFFIX_EMBED_TOKENS: (262144, 1536),
            PREFIX + constants.SUFFIX_EMBED_TOKENS_PER_LAYER: (262144, 35 * 256),
        }.items():
            embedding = _module(model, path)
            if not isinstance(embedding, (nn.Embedding, QuantizedEmbedding)):
                _layout(path)
            if (embedding.num_embeddings, embedding.embedding_dim) != dimensions:
                _layout(path)
        if (model.lm_head.in_features, model.lm_head.out_features) != (1536, 262144):
            _layout(constants.PATH_LM_HEAD)
        if type(decoder.norm) is not g.Gemma4RMSNorm:
            _layout(PREFIX + constants.SUFFIX_NORM)
        # Reject custom text modules and instance-level forward replacements.
        allowed = {
            value for value in vars(g).values()
            if isinstance(value, type) and issubclass(value, nn.Module)
            and value.__module__ == g.__name__
        } | {nn.ModuleList, nn.Linear, nn.Embedding, QuantizedLinear, QuantizedEmbedding}
        from transformers.activations import ACT2FN
        allowed.add(type(ACT2FN[c.hidden_activation]))
        for path, module in decoder.named_modules():
            if type(module) not in allowed or constants.METHOD_FORWARD in vars(module):
                _layout(constants.TEMPLATE_DOTTED_PAIR.format(PREFIX, path).rstrip(constants.DOT))
        if (
            constants.METHOD_FORWARD in vars(model) or constants.METHOD_FORWARD in vars(model.lm_head)
            or type(model.lm_head) not in (nn.Linear, QuantizedLinear)
        ):
            _layout(constants.PATH_LM_HEAD)
        modules = inspector.inspect(model, config.max_modules)
        layers: list[LayerInfo] = []
        producers: dict[str, int] = {}
        rotary_parameters = c.rope_parameters
        if rotary_parameters is None:
            _layout(constants.PATH_ROPE_PARAMETERS)
        for i, layer in enumerate(decoder.layers):
            path = constants.TEMPLATE_LAYER_PATH.format(PREFIX, i)
            if not isinstance(layer, g.Gemma4TextDecoderLayer):
                _layout(path)
            if type(layer.mlp) is not g.Gemma4TextMLP:
                _layout(path + constants.SUFFIX_MLP)
            a = layer.self_attn
            if type(a) is not g.Gemma4TextAttention or a.layer_type != expected_types[i]:
                _layout(path + constants.SUFFIX_SELF_ATTN)
            attention_type = expected_types[i]
            shared = i >= 15
            width = 12288 if shared else 6144
            head = 512 if i % 5 == 4 else 256
            if a.head_dim != head or a.is_kv_shared_layer != shared or layer.mlp.intermediate_size != width:
                _layout(path)
            if not a.is_causal or a.num_key_value_groups != 8:
                _layout(path + constants.SUFFIX_SELF_ATTN)
            for norm in (
                constants.PATH_INPUT_NORM, constants.PATH_POST_ATTN_NORM,
                constants.PATH_PRE_FF_NORM, constants.PATH_POST_FF_NORM,
                constants.PATH_POST_PLE_NORM, constants.PATH_Q_NORM,
                *(() if shared else (constants.PATH_K_NORM, constants.PATH_V_NORM)),
            ):
                normalization = _module(layer, norm)
                if type(normalization) is not g.Gemma4RMSNorm:
                    _layout(constants.TEMPLATE_DOTTED_PAIR.format(path, norm))
                if normalization.eps != c.rms_norm_eps:
                    _layout(constants.TEMPLATE_NORM_EPS_PATH.format(path, norm))
            required = {
                constants.PATH_Q_PROJ: (1536, 8 * head),
                constants.PATH_O_PROJ: (8 * head, 1536),
                constants.PATH_MLP_GATE: (1536, width), constants.PATH_MLP_UP: (1536, width),
                constants.PATH_MLP_DOWN: (width, 1536),
                constants.PATH_PLE_GATE: (1536, 256),
                constants.PATH_PLE_PROJECTION: (256, 1536),
            }
            if shared:
                if hasattr(a, constants.ATTR_K_PROJ) or hasattr(a, constants.ATTR_V_PROJ):
                    _layout(path + constants.SUFFIX_SELF_ATTN)
            else:
                required.update({
                    constants.PATH_K_PROJ: (1536, head),
                    constants.PATH_V_PROJ: (1536, head),
                })
            for relative, dimensions in required.items():
                m = _module(layer, relative)
                if type(m) not in (nn.Linear, QuantizedLinear) or (m.in_features, m.out_features) != dimensions:
                    _layout(constants.TEMPLATE_DOTTED_PAIR.format(path, relative))
            if not shared and a.store_full_length_kv:
                producers[attention_type] = i
            if shared and attention_type not in producers:
                _layout(path + constants.SUFFIX_SHARED_KV)
            rotary = rotary_parameters[attention_type]
            layers.append(LayerInfo(
                i, path, attention_type, 1536, head, 8, 1, width,
                producers[attention_type] if shared else i, shared,
                a.sliding_window, c.hidden_activation,
                float(rotary[constants.FIELD_ROPE_THETA]), float(rotary.get(constants.FIELD_PARTIAL_ROTARY_FACTOR, 1.0)),
                float(layer.input_layernorm.eps), float(a.scaling),
            ))
        inventory = tuple(layers)
        decoders = tuple(self._decoder(layer, model) for layer in inventory)
        hierarchy = _Graph(constants.VIEW_HIERARCHY, constants.TEXT_MODULE_CONTAINMENT_NOT_EXECUTION_ORDER)
        for i, item in enumerate(modules):
            hierarchy.add(constants.TEMPLATE_MODULE_NODE_ID.format(i), constants.TEMPLATE_MODULE_LABEL.format(item.path or constants.PATH_MODEL, item.class_name), constants.KIND_MODULE, item.path)
        ids = {item.path: constants.TEMPLATE_MODULE_NODE_ID.format(i) for i, item in enumerate(modules)}
        for item in modules[1:]:
            parent = item.path.rpartition(constants.DOT)[0]
            hierarchy.link(ids[parent], ids[item.path], kind=constants.EDGE_CONTAINS)
        return ArchitectureGraph(
            self._overview(inventory, c.final_logit_softcapping),
            decoders, hierarchy.finish(), modules, inventory
        )

    def _overview(
        self, layers: tuple[LayerInfo, ...], softcap: float | None
    ) -> GraphView:
        b = _Graph(constants.VIEW_OVERVIEW, constants.TEXT_GEMMA_E2B_TEXT_PATH_MODALITY_TOWERS_COLLAPSED)
        b.add(constants.NODE_TOKENS, constants.TEXT_TOKEN_IDS_B_S, constants.KIND_INPUT)
        b.add(constants.NODE_EMBED, constants.TEXT_SCALED_TOKEN_EMBEDDING_262144_X_1536, constants.KIND_EMBEDDING, PREFIX + constants.SUFFIX_EMBED_TOKENS)
        b.add(constants.NODE_PLE, constants.TEXT_PER_LAYER_INPUTS_TOKEN_PLE_NORMALIZED_CONTEXT_PROJECTION_SCALE_1_SQRT_2, constants.KIND_OPERATION)
        b.add(constants.NODE_POSITIONS, constants.TEXT_ROPE_DISTINCT_SLIDING_FULL_POSITIONAL_EMBEDDINGS, constants.KIND_POSITION)
        b.link(constants.NODE_TOKENS, constants.NODE_EMBED)
        b.link(constants.NODE_TOKENS, constants.NODE_PLE)
        b.link(constants.NODE_EMBED, constants.NODE_PLE)
        prev = constants.NODE_EMBED
        for layer in layers:
            key = constants.TEMPLATE_LAYER_NODE_ID.format(layer.index)
            b.add(key, constants.TEMPLATE_BLOCK_LABEL.format(layer.index, layer.attention_type, layer.head_dim, layer.intermediate_size)
                  + (constants.TEMPLATE_KV_SOURCE_LABEL.format(layer.kv_source) if layer.shared_kv else constants.TEXT_OWN_KV), constants.VIEW_DECODER, layer.path)
            b.link(prev, key)
            b.link(constants.NODE_PLE, key)
            b.link(constants.NODE_POSITIONS, key)
            if layer.shared_kv:
                b.link(constants.TEMPLATE_LAYER_NODE_ID.format(layer.kv_source), key, kind=constants.EDGE_SHARED_STATE)
            prev = key
        b.add(constants.NODE_NORM, constants.TEXT_FINAL_RMSNORM_1536, constants.KIND_NORMALIZATION, PREFIX + constants.SUFFIX_NORM)
        b.add(constants.NODE_HEAD, constants.TEMPLATE_LM_HEAD_LABEL.format(softcap), constants.KIND_LINEAR, constants.PATH_LM_HEAD)
        b.add(constants.NODE_OUT, constants.TEXT_VOCABULARY_LOGITS_B_S_262144, constants.KIND_OUTPUT)
        b.link(prev, constants.NODE_NORM, constants.NODE_HEAD, constants.NODE_OUT)
        b.add(constants.NODE_MODALITIES, constants.TEXT_VISION_AUDIO_TOWERS_COLLAPSED_INTERNALS_AND_FUSION_OUTSIDE_TEXT_SCOPE, constants.KIND_BOUNDARY)
        return b.finish()

    def _decoder(self, layer: LayerInfo, model: nn.Module) -> GraphView:
        b = _Graph(constants.TEMPLATE_DECODER_VIEW_ID.format(layer.index), constants.TEMPLATE_DECODER_TITLE.format(layer.index))
        path = layer.path

        def node(key: str, label: str, kind: str, relative: str | None = None) -> None:
            module_path = constants.TEMPLATE_DOTTED_PAIR.format(path, relative) if relative else None
            if module_path:
                module = model.get_submodule(module_path)
                label += constants.TEMPLATE_CLASS_LABEL_SUFFIX.format(type(module).__name__)
                if isinstance(module, nn.Linear):
                    label += constants.TEMPLATE_LINEAR_DIMENSIONS.format(module.in_features, module.out_features)
                    bits = vars(module).get(constants.FIELD_NUM_BITS)
                    if bits is not None:
                        label += constants.TEMPLATE_PACKED_BITS.format(bits)
                if type(module) is g.Gemma4RMSNorm:
                    label += constants.TEMPLATE_NORM_EPSILON.format(module.eps)
            b.add(key, label, kind, module_path)

        node(constants.KIND_INPUT, constants.TEMPLATE_HIDDEN_STATES.format(layer.hidden_size), constants.KIND_INPUT)
        node(constants.NODE_INPUT_NORM, constants.TEXT_INPUT_RMSNORM, constants.KIND_NORMALIZATION, constants.PATH_INPUT_NORM)
        node(constants.NODE_Q, constants.TEXT_Q_PROJECTION, constants.KIND_LINEAR, constants.PATH_Q_PROJ)
        node(constants.NODE_Q_NORM, constants.TEMPLATE_Q_NORM_ROPE.format(layer.rope_theta, layer.rotary_fraction),
             constants.KIND_NORMALIZATION, constants.PATH_Q_NORM)
        node(constants.NODE_ATTENTION, constants.TEMPLATE_ATTENTION_LABEL.format(layer.attention_type, layer.head_dim, layer.sliding_window, layer.attention_scale), constants.KIND_ATTENTION, constants.PATH_SELF_ATTN)
        node(constants.NODE_O, constants.TEXT_O_PROJECTION, constants.KIND_LINEAR, constants.PATH_O_PROJ)
        node(constants.NODE_POST_ATTN_NORM, constants.TEXT_POST_ATTENTION_RMSNORM, constants.KIND_NORMALIZATION, constants.PATH_POST_ATTN_NORM)
        node(constants.NODE_ATTN_RESIDUAL, constants.TEXT_RESIDUAL_ADD, constants.EDGE_RESIDUAL)
        b.link(constants.KIND_INPUT, constants.NODE_INPUT_NORM, constants.NODE_Q, constants.NODE_Q_NORM, constants.NODE_ATTENTION, constants.NODE_O, constants.NODE_POST_ATTN_NORM, constants.NODE_ATTN_RESIDUAL)
        b.link(constants.KIND_INPUT, constants.NODE_ATTN_RESIDUAL, kind=constants.EDGE_RESIDUAL)
        if layer.shared_kv:
            node(constants.NODE_KV, constants.TEMPLATE_EXTERNAL_KV.format(layer.kv_source), constants.KIND_INPUT)
            b.link(constants.NODE_KV, constants.NODE_ATTENTION, kind=constants.EDGE_SHARED_STATE)
        else:
            node(constants.NODE_K, constants.TEXT_K_PROJECTION, constants.KIND_LINEAR, constants.PATH_K_PROJ)
            node(constants.NODE_V, constants.TEXT_V_PROJECTION, constants.KIND_LINEAR, constants.PATH_V_PROJ)
            node(constants.NODE_K_NORM, constants.TEXT_K_RMSNORM_THEN_ROPE, constants.KIND_NORMALIZATION, constants.PATH_K_NORM)
            node(constants.NODE_V_NORM, constants.TEXT_V_RMSNORM_NO_LEARNED_SCALE, constants.KIND_NORMALIZATION, constants.PATH_V_NORM)
            b.link(constants.NODE_INPUT_NORM, constants.NODE_K, constants.NODE_K_NORM, constants.NODE_ATTENTION)
            b.link(constants.NODE_INPUT_NORM, constants.NODE_V, constants.NODE_V_NORM, constants.NODE_ATTENTION)
        node(constants.NODE_PRE_FF_NORM, constants.TEXT_PRE_MLP_RMSNORM, constants.KIND_NORMALIZATION, constants.PATH_PRE_FF_NORM)
        node(constants.NODE_GATE, constants.TEXT_GATE_PROJECTION, constants.KIND_LINEAR, constants.PATH_MLP_GATE)
        node(constants.NODE_ACTIVATION, layer.activation, constants.KIND_ACTIVATION)
        node(constants.NODE_UP, constants.TEXT_UP_PROJECTION, constants.KIND_LINEAR, constants.PATH_MLP_UP)
        node(constants.NODE_MULTIPLY, constants.TEXT_ELEMENTWISE_MULTIPLY, constants.KIND_OPERATION)
        node(constants.NODE_DOWN, constants.TEXT_DOWN_PROJECTION, constants.KIND_LINEAR, constants.PATH_MLP_DOWN)
        node(constants.NODE_POST_FF_NORM, constants.TEXT_POST_MLP_RMSNORM, constants.KIND_NORMALIZATION, constants.PATH_POST_FF_NORM)
        node(constants.NODE_FF_RESIDUAL, constants.TEXT_RESIDUAL_ADD, constants.EDGE_RESIDUAL)
        b.link(constants.NODE_ATTN_RESIDUAL, constants.NODE_PRE_FF_NORM, constants.NODE_GATE, constants.NODE_ACTIVATION, constants.NODE_MULTIPLY, constants.NODE_DOWN, constants.NODE_POST_FF_NORM, constants.NODE_FF_RESIDUAL)
        b.link(constants.NODE_PRE_FF_NORM, constants.NODE_UP, constants.NODE_MULTIPLY)
        b.link(constants.NODE_ATTN_RESIDUAL, constants.NODE_FF_RESIDUAL, kind=constants.EDGE_RESIDUAL)
        node(constants.NODE_PLE_GATE, constants.TEXT_PER_LAYER_INPUT_GATE, constants.KIND_LINEAR, constants.PATH_PLE_GATE)
        node(constants.NODE_PLE_ACTIVATION, layer.activation, constants.KIND_ACTIVATION)
        node(constants.NODE_PLE_INPUT, constants.TEMPLATE_PER_LAYER_INPUT.format(layer.index), constants.KIND_INPUT)
        node(constants.NODE_PLE_MULTIPLY, constants.TEXT_ELEMENTWISE_MULTIPLY, constants.KIND_OPERATION)
        node(constants.NODE_PLE_PROJECTION, constants.TEXT_PER_LAYER_PROJECTION, constants.KIND_LINEAR, constants.PATH_PLE_PROJECTION)
        node(constants.NODE_PLE_NORM, constants.TEXT_POST_PER_LAYER_RMSNORM, constants.KIND_NORMALIZATION, constants.PATH_POST_PLE_NORM)
        node(constants.NODE_PLE_RESIDUAL, constants.TEXT_RESIDUAL_ADD, constants.EDGE_RESIDUAL)
        node(constants.NODE_SCALE, constants.TEXT_MULTIPLY_LEARNED_LAYER_SCALAR_VALUE_NOT_INSPECTED, constants.KIND_OPERATION)
        node(constants.KIND_OUTPUT, constants.TEXT_OUTPUT_B_S_1536, constants.KIND_OUTPUT)
        b.link(constants.NODE_FF_RESIDUAL, constants.NODE_PLE_GATE, constants.NODE_PLE_ACTIVATION, constants.NODE_PLE_MULTIPLY, constants.NODE_PLE_PROJECTION, constants.NODE_PLE_NORM, constants.NODE_PLE_RESIDUAL, constants.NODE_SCALE, constants.KIND_OUTPUT)
        b.link(constants.NODE_PLE_INPUT, constants.NODE_PLE_MULTIPLY)
        b.link(constants.NODE_FF_RESIDUAL, constants.NODE_PLE_RESIDUAL, kind=constants.EDGE_RESIDUAL)
        return b.finish((layer.index,))
