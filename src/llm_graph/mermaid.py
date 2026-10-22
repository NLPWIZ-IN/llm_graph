import re

from . import constants
from .config import DiagramConfig
from .errors import InvalidConfigurationError, UnsupportedViewError
from .schema import DiagramArtifact, GraphView


def escape_label(value: str) -> str:
    # Mermaid decimal entities keep user/model text out of diagram syntax.
    return constants.EMPTY.join(
        c if c.isascii() and (c.isalnum() or c in constants.MERMAID_SAFE_LABEL_CHARACTERS) else constants.TEMPLATE_MERMAID_ENTITY.format(ord(c))
        for c in value
    )


class MermaidBackend:
    name = constants.BACKEND_MERMAID

    def validate(self, config: DiagramConfig) -> None:
        if config.format not in (constants.FORMAT_MARKDOWN, constants.BACKEND_MERMAID):
            raise InvalidConfigurationError(
                constants.ERROR_UNSUPPORTED_FORMAT, constants.STAGE_VALIDATION, backend=self.name, format=config.format
            )
        for view in config.views:
            if view == constants.VIEW_EXECUTION or config.scope != constants.SCOPE_TEXT:
                raise UnsupportedViewError(
                    constants.ERROR_UNSUPPORTED_VIEW, constants.STAGE_VALIDATION, backend=self.name, adapter=constants.GEMMA_ADAPTER,
                    view=view, scope=config.scope,
                )

    def render(
        self, views: tuple[GraphView, ...], config: DiagramConfig
    ) -> tuple[DiagramArtifact, ...]:
        result: list[DiagramArtifact] = []
        for view in views:
            identifiers = [view.id, *(n.id for n in view.nodes)]
            if any(re.fullmatch(constants.MERMAID_IDENTIFIER_PATTERN, v) is None for v in identifiers):
                raise InvalidConfigurationError(
                    constants.ERROR_INVALID_CONFIGURATION, constants.STAGE_RENDER, field=constants.FIELD_GRAPH_ID, reason=constants.TEXT_UNSAFE_IDENTIFIER
                )
            lines = [constants.TEMPLATE_MERMAID_FLOWCHART.format(config.direction)]
            lines.extend(constants.TEMPLATE_MERMAID_NODE.format(n.id, escape_label(n.kind + constants.COLON_SPACE + n.label)) for n in view.nodes)
            operators = {
                constants.EDGE_DATA_FLOW: constants.MERMAID_DATA_ARROW, constants.EDGE_RESIDUAL: constants.MERMAID_RESIDUAL_ARROW,
                constants.EDGE_SHARED_STATE: constants.MERMAID_SHARED_KV_ARROW, constants.EDGE_CONTAINS: constants.MERMAID_CONTAINS_ARROW,
            }
            for edge in view.edges:
                if edge.kind not in operators:
                    raise InvalidConfigurationError(
                        constants.ERROR_INVALID_CONFIGURATION, constants.STAGE_RENDER, field=constants.FIELD_EDGE_KIND, reason=constants.TEXT_UNSUPPORTED_EDGE_KIND
                    )
                lines.append(constants.TEMPLATE_MERMAID_EDGE.format(edge.source, operators[edge.kind], edge.target))
            content = constants.NEWLINE.join(lines) + constants.NEWLINE
            if config.format == constants.FORMAT_MARKDOWN:
                content = (
                    constants.TEMPLATE_MERMAID_SECTION.format(escape_label(view.title), content)
                )
            result.append(DiagramArtifact(
                constants.TEMPLATE_DOTTED_PAIR.format(view.id, constants.EXTENSION_MARKDOWN if config.format == constants.FORMAT_MARKDOWN else constants.EXTENSION_MERMAID),
                content, constants.MEDIA_MARKDOWN if config.format == constants.FORMAT_MARKDOWN else constants.MEDIA_MERMAID,
            ))
        return tuple(result)
