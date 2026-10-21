from dataclasses import dataclass

from . import constants
from .errors import InvalidConfigurationError


@dataclass(frozen=True)
class DiagramConfig:
    backend: str = constants.BACKEND_MERMAID
    format: str = constants.FORMAT_MARKDOWN
    views: tuple[str, ...] = (constants.VIEW_OVERVIEW, constants.VIEW_VARIANTS)
    layer_index: int = 0
    scope: str = constants.SCOPE_TEXT
    max_nodes: int = 400
    max_modules: int = 10000
    direction: str = constants.DIRECTION_TOP_DOWN

    def __post_init__(self) -> None:
        def invalid(field: str, reason: str) -> None:
            raise InvalidConfigurationError(
                constants.ERROR_INVALID_CONFIGURATION, constants.STAGE_VALIDATION, field=field, reason=reason
            )

        for name in (constants.FIELD_BACKEND, constants.FIELD_FORMAT, constants.FIELD_SCOPE, constants.FIELD_DIRECTION):
            if not isinstance(getattr(self, name), str):
                invalid(name, constants.TEXT_MUST_BE_A_STRING)
        for name in (constants.FIELD_LAYER_INDEX, constants.FIELD_MAX_NODES, constants.FIELD_MAX_MODULES):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == constants.FIELD_LAYER_INDEX else 1):
                invalid(name, constants.TEXT_MUST_BE_AN_INTEGER_WITHIN_THE_DOCUMENTED_RANGE)
        if self.direction not in (constants.DIRECTION_TOP_DOWN, constants.DIRECTION_LEFT_RIGHT, constants.DIRECTION_BOTTOM_UP, constants.DIRECTION_RIGHT_LEFT):
            invalid(constants.FIELD_DIRECTION, constants.TEXT_EXPECTED_TD_LR_BT_OR_RL)
        if not isinstance(self.views, tuple) or not self.views:
            invalid(constants.FIELD_VIEWS, constants.TEXT_EXPECTED_A_NONEMPTY_TUPLE)
        if any(not isinstance(v, str) or v not in (constants.VIEW_OVERVIEW, constants.VIEW_DECODER, constants.VIEW_VARIANTS, constants.VIEW_HIERARCHY, constants.VIEW_EXECUTION) for v in self.views):
            invalid(constants.FIELD_VIEWS, constants.TEXT_UNKNOWN_VIEW)
        if len(set(self.views)) != len(self.views):
            invalid(constants.FIELD_VIEWS, constants.TEXT_DUPLICATE_VIEWS)
