from dataclasses import replace

from . import constants
from .config import DiagramConfig
from .errors import InvalidGraphError, InvalidConfigurationError, RenderLimitError
from .schema import ArchitectureGraph, GraphView


class ViewBuilder:
    def build(
        self, graph: ArchitectureGraph, selection: DiagramConfig
    ) -> tuple[GraphView, ...]:
        result: list[GraphView] = []
        for kind in selection.views:
            if kind == constants.VIEW_OVERVIEW:
                result.append(graph.overview)
            elif kind == constants.VIEW_HIERARCHY:
                result.append(graph.hierarchy)
            elif kind == constants.VIEW_DECODER:
                if selection.layer_index >= len(graph.layers):
                    raise InvalidConfigurationError(
                        constants.ERROR_INVALID_CONFIGURATION, constants.STAGE_VIEW, field=constants.FIELD_LAYER_INDEX, reason=constants.TEXT_OUTSIDE_DECODER_RANGE
                    )
                result.append(graph.decoders[selection.layer_index])
            elif kind == constants.VIEW_VARIANTS:
                groups: dict[tuple[object, ...], list[int]] = {}
                for layer in graph.layers:
                    groups.setdefault(layer.signature, []).append(layer.index)
                for i, indices in enumerate(groups.values()):
                    representative = graph.decoders[indices[0]]
                    result.append(replace(
                        representative, id=constants.TEMPLATE_VARIANT_VIEW_ID.format(i),
                        title=constants.TEMPLATE_VARIANT_TITLE.format(i, indices[0], indices),
                        member_indices=tuple(indices),
                    ))
        for view in result:
            ids = [n.id for n in view.nodes]
            if len(set(ids)) != len(ids) or any(
                e.source not in ids or e.target not in ids for e in view.edges
            ):
                raise InvalidGraphError(
                    constants.ERROR_INVALID_GRAPH, constants.STAGE_VIEW, invariant=constants.TEXT_UNIQUE_IDS_AND_VALID_ENDPOINTS,
                    component_path=view.id,
                )
            if len(ids) > selection.max_nodes:
                raise RenderLimitError(
                    constants.ERROR_RENDER_LIMIT, constants.STAGE_VIEW, view=view.id,
                    node_count=len(ids), limit=selection.max_nodes,
                )
        return tuple(result)
