from torch import nn

from . import constants
from .errors import InspectionLimitError
from .schema import ModuleInfo, TensorInfo


class ModuleInspector:
    def inspect(self, model: nn.Module, limit: int) -> tuple[ModuleInfo, ...]:
        result: list[ModuleInfo] = []
        seen: dict[int, str] = {}
        pending = [(constants.EMPTY, model)]
        allowed = (
            constants.FIELD_IN_FEATURES, constants.FIELD_OUT_FEATURES, constants.FIELD_NUM_EMBEDDINGS, constants.FIELD_EMBEDDING_DIM,
            constants.FIELD_NUM_BITS, constants.FIELD_EPS, constants.FIELD_SCALAR_EMBED_SCALE, constants.FIELD_INTERMEDIATE_SIZE,
            constants.FIELD_HEAD_DIM, constants.FIELD_IS_CAUSAL, constants.FIELD_IS_SLIDING, constants.FIELD_SLIDING_WINDOW,
            constants.FIELD_NUM_KV_GROUPS, constants.FIELD_ATTENTION_DROPOUT, constants.FIELD_SCALING,
        )
        while pending:
            path, module = pending.pop()
            if len(result) >= limit:
                raise InspectionLimitError(
                    constants.ERROR_INSPECTION_LIMIT, constants.STAGE_INSPECTION, limit_name=constants.FIELD_MAX_MODULES, limit=limit
                )
            alias = seen.get(id(module))
            properties: dict[str, str | int | float | bool] = {}
            for name in allowed:
                value = vars(module).get(name)
                if isinstance(value, (str, int, float, bool)):
                    properties[name] = value
            if constants.FIELD_BIAS in module._parameters:
                properties[constants.FIELD_BIAS] = module._parameters[constants.FIELD_BIAS] is not None
            # Read registered tensors, never descriptors such as quantized embedding.weight.
            tensors = tuple(
                TensorInfo(name, tuple(t.shape), str(t.dtype), str(t.device))
                for name, t in (
                    *module.named_parameters(recurse=False),
                    *module.named_buffers(recurse=False),
                )
            )
            result.append(ModuleInfo(path, type(module).__name__, properties, tensors, alias))
            if alias is not None:
                continue
            seen[id(module)] = path
            pending.extend(
                (constants.TEMPLATE_DOTTED_PAIR.format(path, name).strip(constants.DOT), child)
                for name, child in reversed(list(module._modules.items()))
                if child is not None
            )
        return tuple(result)
