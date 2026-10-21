import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from torch import nn

from . import constants
from .config import DiagramConfig
from .contracts import ArchitectureAdapter, DiagramBackend
from .errors import (
    ArtifactWriteError, Diagnostic, InvalidConfigurationError,
    InvalidModelInputError, UnsupportedArchitectureError, InspectionError, AmbiguousAdapterError,
)
from .gemma4 import Gemma4Adapter, configured_bits
from .inspection import ModuleInspector
from .mermaid import MermaidBackend
from .schema import ArchitectureGraph, DiagramArtifact
from .views import ViewBuilder


@dataclass(frozen=True)
class VisualizationReport:
    graph: ArchitectureGraph
    artifacts: tuple[DiagramArtifact, ...]
    diagnostics: tuple[Diagnostic, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            constants.FIELD_SCHEMA_VERSION: 1, **asdict(self),
            constants.FIELD_PROPERTY_SOURCES: {
                m.path: m.property_sources for m in self.graph.modules
            },
            constants.FIELD_TENSOR_METADATA_SOURCE: constants.SOURCE_REGISTERED_TENSORS,
            constants.FIELD_TOPOLOGY_SOURCE: constants.SOURCE_VERIFIED_GEMMA_FORWARD,
        }

    def markdown(self) -> str:
        """Combine Mermaid artifacts and a concise layer inventory."""
        parts = [
            constants.REPORT_HEADING,
            constants.REPORT_PROVENANCE,
        ]
        for artifact in self.artifacts:
            if artifact.media_type == constants.MEDIA_MARKDOWN:
                parts.append(artifact.content)
            else:
                parts.append(constants.TEMPLATE_MERMAID_FENCE.format(artifact.content))
        inventory = [
            constants.REPORT_INVENTORY_HEADER
        ]
        for layer in self.graph.layers:
            inventory.append(
                constants.TEMPLATE_INVENTORY_ROW.format(layer.index, layer.attention_type, layer.head_dim, layer.query_heads, layer.kv_heads, layer.intermediate_size, layer.kv_source)
            )
        parts.append(constants.EMPTY.join(inventory))
        parts.append(constants.REPORT_DIAGNOSTICS_HEADER)
        parts.extend(constants.TEMPLATE_DIAGNOSTIC_BULLET.format(d.code, d.message) for d in self.diagnostics)
        return constants.NEWLINE.join(parts)

    def write(self, directory: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
        destination = Path(directory)
        outputs = (
            DiagramArtifact(constants.REPORT_MARKDOWN, self.markdown(), constants.MEDIA_MARKDOWN),
            DiagramArtifact(constants.REPORT_JSON, json.dumps(self.to_dict(), indent=2), constants.MEDIA_JSON),
            *self.artifacts,
        )
        if len({a.name for a in outputs}) != len(outputs):
            raise InvalidConfigurationError(constants.ERROR_UNSAFE_ARTIFACT_NAME, constants.STAGE_WRITE, artifact_name=constants.TEXT_DUPLICATE_NAMES)
        for artifact in outputs:
            if Path(artifact.name).name != artifact.name or artifact.name in (constants.EMPTY, constants.DOT, constants.PARENT_DIRECTORY):
                raise InvalidConfigurationError(constants.ERROR_UNSAFE_ARTIFACT_NAME, constants.STAGE_WRITE, artifact_name=artifact.name)
            if (destination / artifact.name).exists() and not overwrite:
                raise ArtifactWriteError(constants.ERROR_ARTIFACT_EXISTS, constants.STAGE_WRITE, artifact_name=artifact.name)
        written: list[Path] = []
        current = constants.OUTPUT_DIRECTORY
        temporary: str | None = None
        try:
            destination.mkdir(parents=True, exist_ok=True)
            for artifact in outputs:
                current = artifact.name
                target = destination / current
                fd, temporary = tempfile.mkstemp(dir=destination, prefix=constants.TEMP_FILE_PREFIX)
                with os.fdopen(fd, constants.FILE_MODE_WRITE, encoding=constants.ENCODING_UTF8) as stream:
                    stream.write(artifact.content)
                if overwrite:
                    os.replace(temporary, target)
                else:
                    # Exclusive creation without overwriting a file created after preflight.
                    os.link(temporary, target)
                    os.unlink(temporary)
                temporary = None
                written.append(target)
        except FileExistsError as exc:
            raise ArtifactWriteError(
                constants.ERROR_ARTIFACT_EXISTS, constants.STAGE_WRITE, artifact_name=current,
                committed_artifacts=constants.COMMA_SPACE.join(p.name for p in written),
            ) from exc
        except OSError as exc:
            raise ArtifactWriteError(
                constants.ERROR_ARTIFACT_WRITE, constants.STAGE_WRITE, artifact_name=current,
                committed_artifacts=constants.COMMA_SPACE.join(p.name for p in written),
            ) from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        return tuple(written)


class AdapterFactory:
    def __init__(self, adapters: Sequence[ArchitectureAdapter]) -> None:
        self.adapters = tuple(adapters)

    def resolve(self, model: nn.Module) -> ArchitectureAdapter:
        matches = [a for a in self.adapters if a.matches(model)]
        if not matches:
            raise UnsupportedArchitectureError(
                constants.ERROR_UNSUPPORTED_ARCHITECTURE, constants.STAGE_RESOLUTION, model_type=type(model).__name__
            )
        if len(matches) > 1:
            raise AmbiguousAdapterError(
                constants.ERROR_AMBIGUOUS_ADAPTER, constants.STAGE_RESOLUTION, model_type=type(model).__name__
            )
        return matches[0]


class BackendFactory:
    def __init__(self, backends: Sequence[DiagramBackend]) -> None:
        self.backends = tuple(backends)
        if len({b.name for b in self.backends}) != len(self.backends):
            raise InvalidConfigurationError(
                constants.ERROR_INVALID_CONFIGURATION, constants.STAGE_RESOLUTION, field=constants.FIELD_BACKENDS, reason=constants.TEXT_DUPLICATE_REGISTRATIONS
            )

    def resolve(self, name: str) -> DiagramBackend:
        for backend in self.backends:
            if backend.name == name:
                return backend
        raise InvalidConfigurationError(constants.ERROR_UNKNOWN_BACKEND, constants.STAGE_RESOLUTION, backend=name)


class ModelVisualizer:
    def __init__(
        self, adapters: AdapterFactory, backends: BackendFactory,
        inspector: ModuleInspector | None = None, view_builder: ViewBuilder | None = None,
    ) -> None:
        self.adapters = adapters
        self.backends = backends
        self.inspector = inspector or ModuleInspector()
        self.view_builder = view_builder or ViewBuilder()

    def visualize(
        self, model: nn.Module, config: DiagramConfig | None = None
    ) -> VisualizationReport:
        config = config or DiagramConfig()
        if not isinstance(model, nn.Module):
            raise InvalidModelInputError(
                constants.ERROR_INVALID_MODEL_INPUT, constants.STAGE_VALIDATION, input_type=type(model).__name__
            )
        backend = self.backends.resolve(config.backend)
        backend.validate(config)
        adapter = self.adapters.resolve(model)
        graph = adapter.inspect(model, self.inspector, config)
        modules = []
        diagnostics = [Diagnostic(
            constants.INFO_COLLAPSED_MODALITY, constants.MESSAGE_COLLAPSED_MODALITY,
            constants.PATH_MODEL,
        )]
        for module in graph.modules:
            properties = dict(module.properties)
            bits = configured_bits(model, module.path)
            if bits is not None:
                properties[constants.FIELD_CONFIGURED_BITS] = bits
                if constants.FIELD_NUM_BITS in properties and properties[constants.FIELD_NUM_BITS] != bits:
                    raise InspectionError(
                        constants.ERROR_PROPERTY_CONFLICT, constants.STAGE_INSPECTION, property=constants.FIELD_NUM_BITS,
                        component_path=module.path,
                    )
                if constants.FIELD_NUM_BITS not in properties:
                    diagnostics.append(Diagnostic(
                        constants.WARNING_CONFIGURED_QUANTIZATION,
                        constants.TEMPLATE_CONFIGURED_QUANTIZATION_WARNING.format(module.path),
                        module.path, constants.SEVERITY_WARNING,
                    ))
            modules.append(replace(module, properties=properties))
        graph = replace(graph, modules=tuple(modules))
        views = self.view_builder.build(graph, config)
        return VisualizationReport(graph, backend.render(views, config), tuple(diagnostics))


def create_visualizer(
    *, adapters: Sequence[ArchitectureAdapter] | None = None,
    backends: Sequence[DiagramBackend] | None = None,
) -> ModelVisualizer:
    return ModelVisualizer(
        AdapterFactory((Gemma4Adapter(),) if adapters is None else adapters),
        BackendFactory((MermaidBackend(),) if backends is None else backends),
    )
