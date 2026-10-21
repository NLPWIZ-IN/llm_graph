from typing import Protocol

from torch import nn

from .config import DiagramConfig
from .inspection import ModuleInspector
from .schema import ArchitectureGraph, DiagramArtifact, GraphView


class ArchitectureAdapter(Protocol):
    name: str

    def matches(self, model: nn.Module) -> bool: ...

    def inspect(
        self, model: nn.Module, inspector: ModuleInspector, config: DiagramConfig
    ) -> ArchitectureGraph: ...


class DiagramBackend(Protocol):
    name: str

    def validate(self, config: DiagramConfig) -> None: ...

    def render(
        self, views: tuple[GraphView, ...], config: DiagramConfig
    ) -> tuple[DiagramArtifact, ...]: ...
