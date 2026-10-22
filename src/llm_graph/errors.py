"""Stable invocation failures and diagnostics for the Gemma release."""

from dataclasses import dataclass
from types import MappingProxyType

from . import constants


@dataclass(frozen=True)
class IssueDefinition:
    template: str
    remediation: str


CATALOG = MappingProxyType({
    constants.ERROR_INVALID_MODEL_INPUT: IssueDefinition(constants.MESSAGE_INVALID_MODEL_INPUT, constants.TEXT_PASS_A_LOADED_LOCAL_PYTORCH_MODEL),
    constants.ERROR_UNSUPPORTED_ARCHITECTURE: IssueDefinition(constants.MESSAGE_UNSUPPORTED_ARCHITECTURE, constants.TEXT_USE_THE_SUPPORTED_GEMMA_CHECKPOINT),
    constants.ERROR_UNSUPPORTED_TRANSFORMERS_VERSION: IssueDefinition(constants.MESSAGE_UNSUPPORTED_TRANSFORMERS_VERSION, constants.TEXT_USE_TRANSFORMERS_5_19_0),
    constants.ERROR_UNSUPPORTED_VARIANT: IssueDefinition(constants.MESSAGE_UNSUPPORTED_VARIANT, constants.TEXT_USE_THE_SUPPORTED_GEMMA_E2B_CONFIGURATION),
    constants.ERROR_AMBIGUOUS_ADAPTER: IssueDefinition(constants.MESSAGE_AMBIGUOUS_ADAPTER, constants.TEXT_INJECT_UNAMBIGUOUS_ADAPTER_REGISTRATIONS),
    constants.ERROR_ARCHITECTURE_LAYOUT: IssueDefinition(constants.MESSAGE_ARCHITECTURE_LAYOUT, constants.TEXT_USE_THE_UNMODIFIED_SUPPORTED_IMPLEMENTATION),
    constants.ERROR_INVALID_CONFIGURATION: IssueDefinition(constants.MESSAGE_INVALID_CONFIGURATION, constants.TEXT_CORRECT_THE_CONFIGURATION),
    constants.ERROR_UNKNOWN_BACKEND: IssueDefinition(constants.MESSAGE_UNKNOWN_BACKEND, constants.TEXT_USE_THE_MERMAID_BACKEND_IN_THIS_RELEASE),
    constants.ERROR_UNSUPPORTED_FORMAT: IssueDefinition(constants.MESSAGE_UNSUPPORTED_FORMAT, constants.TEXT_USE_MERMAID_OR_MARKDOWN),
    constants.ERROR_UNSUPPORTED_VIEW: IssueDefinition(constants.MESSAGE_UNSUPPORTED_VIEW, constants.TEXT_USE_OVERVIEW_DECODER_VARIANTS_OR_HIERARCHY_WITH_TEXT_SCOPE),
    constants.ERROR_INSPECTION_LIMIT: IssueDefinition(constants.MESSAGE_INSPECTION_LIMIT, constants.TEXT_INCREASE_THE_INSPECTION_LIMIT_DELIBERATELY),
    constants.ERROR_INVALID_GRAPH: IssueDefinition(constants.MESSAGE_INVALID_GRAPH, constants.TEXT_CORRECT_THE_ADAPTER_GRAPH),
    constants.ERROR_PROPERTY_CONFLICT: IssueDefinition(constants.MESSAGE_PROPERTY_CONFLICT, constants.TEXT_RESOLVE_THE_INCOMPATIBLE_QUANTIZATION_CONFIGURATION),
    constants.ERROR_RENDER_LIMIT: IssueDefinition(constants.MESSAGE_RENDER_LIMIT, constants.TEXT_SELECT_A_SMALLER_VIEW_OR_RAISE_MAX_NODES),
    constants.ERROR_ARTIFACT_EXISTS: IssueDefinition(constants.MESSAGE_ARTIFACT_EXISTS, constants.TEXT_SELECT_ANOTHER_DIRECTORY_OR_ENABLE_OVERWRITE),
    constants.ERROR_ARTIFACT_WRITE: IssueDefinition(constants.MESSAGE_ARTIFACT_WRITE, constants.TEXT_CHECK_DESTINATION_PERMISSIONS_AND_DISK_SPACE),
    constants.ERROR_UNSAFE_ARTIFACT_NAME: IssueDefinition(constants.MESSAGE_UNSAFE_ARTIFACT_NAME, constants.TEXT_USE_A_SAFE_RELATIVE_FILENAME),
})


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    component_path: str | None = None
    severity: str = constants.SEVERITY_INFO


class LLMGraphError(Exception):
    def __init__(self, code: str, stage: str, **details: str | int) -> None:
        definition = CATALOG[code]
        self.code = code
        self.stage = stage
        self.details = {
            key: value[:200] if isinstance(value, str) else value
            for key, value in details.items()
        }
        self.message = definition.template.format(**self.details)
        self.remediation = definition.remediation
        self.component_path = self.details.get(constants.FIELD_COMPONENT_PATH)
        super().__init__(constants.TEMPLATE_EXCEPTION.format(code, self.message))

    def to_dict(self) -> dict[str, object]:
        return {
            constants.FIELD_ISSUE_SCHEMA_VERSION: 1, constants.FIELD_CODE: self.code, constants.FIELD_SEVERITY: constants.SEVERITY_ERROR,
            constants.FIELD_MESSAGE: self.message, constants.FIELD_STAGE: self.stage,
            constants.FIELD_COMPONENT_PATH: self.component_path, constants.FIELD_DETAILS: dict(self.details),
            constants.FIELD_REMEDIATION: self.remediation,
        }


class InvalidModelInputError(LLMGraphError):
    pass


class UnsupportedArchitectureError(LLMGraphError):
    pass


class AmbiguousAdapterError(UnsupportedArchitectureError):
    pass


class InvalidConfigurationError(LLMGraphError):
    pass


class UnsupportedViewError(InvalidConfigurationError):
    pass


class InspectionError(LLMGraphError):
    pass


class InspectionLimitError(InspectionError):
    pass


class InvalidGraphError(InspectionError):
    pass


class RenderLimitError(LLMGraphError):
    pass


class ArtifactWriteError(LLMGraphError):
    pass
