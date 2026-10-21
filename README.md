# llm-graph

**Inspect a supported LLM's architecture without running inference, then generate
Mermaid diagrams and structured layer metadata for technical documentation.**

`llm-graph` is the distribution name; `llm_graph` is the Python import.
This alpha library turns an already loaded PyTorch/Transformers model into a
read-only architecture report. Architecture-specific adapters interpret supported
module layouts, a backend-neutral graph captures the findings, and a Mermaid
backend produces diagrams suitable for Markdown documentation.

## Objectives

- Make model internals understandable through an overview, decoder-block views,
  structural variants, and a precise module inventory.
- Preserve architectural distinctions such as attention windows, shared KV
  states, normalization placement, gated MLPs, and residual connections.
- Distinguish logical dimensions from quantized storage shapes, and configured
  bit widths from inspected packed-module properties.
- Make documentation reproducible through deterministic diagrams, JSON metadata,
  provenance, and stable error codes.
- Support new architectures and renderers through small interfaces, injected
  factories, and composition rather than provider-specific logic in callers.

## Intended audience and use cases

For ML engineers, researchers, technical writers, and learners documenting or
exploring a supported local model. Useful for architecture walkthroughs,
quantization inspection, reproducible model notes, and comparing structural
decoder variants within a model.

This is **not** an inference engine, performance profiler, arbitrary computation
tracer, memory benchmark, or universal model visualizer. It cannot recover
proprietary internal layers from Gemini or Claude API clients.

## Current release and goals

The initial implementation supports only the text architecture of
`google/gemma-4-E2B-it-qat-mobile-transformers`. The extension architecture is
broader than the delivered compatibility matrix; unsupported models fail rather
than receiving a guessed Transformer diagram.

Near-term goals are broader fixture coverage, a documented compatibility
matrix across tested dependency versions, and stronger release automation.
Future candidates include additional verified adapters, an explicitly labeled
generic module hierarchy, and opt-in execution visualization. These are roadmap
items, **not features available in this release**. See [CONTRIBUTING.md](./CONTRIBUTING.md)
for extension requirements and [Release policy](./PUBLISHING.md) for versioning
and compatibility commitments.


## Supported first release

- Python 3.14, Transformers **5.19.0**, PyTorch 2.14.1 (tested environment).
- The checkpoint's `Gemma4ForConditionalGeneration` wrapper and E2B text
  architecture, including native `QuantizedLinear`/`QuantizedEmbedding` modules.
- Mermaid overview, decoder variants, selected decoder, and module hierarchy.
- Logical dimensions, packed tensor shapes, registered dtype/device metadata,
  normalization epsilon, activation, RoPE properties and configured/actual
  module bit widths in the JSON inventory.
- Cross-layer KV sharing, sliding/full attention, doubled MLP widths, per-layer
  inputs, residual paths and layer scaling.
- Explicit file writing and stable coded failures.

Vision/audio towers are inventoried but their internal execution/fusion is **not**
diagrammed. This is not a checkpoint identity/checksum verifier: acceptance is
based on concrete implementation and the E2B architectural profile, not a trusted
model name. No model download, inference, dequantization, tensor-value access,
device movement, or automatic output writes occur during visualization.

Llama, generic unknown-model fallback, opaque provider-client classification and
Torchview execution are **deferred**. Unknown models fail explicitly; Torchview
is not registered. The broader [design](./DESIGN.md) describes future extensions,
not capabilities already delivered.

## Installation

Requires Python 3.14 or later, PyTorch 2.14.1 or later, and Transformers 5.19.0.
Architecture verification currently covers only the tested versions listed
above; broader installation constraints do not imply broader adapter support.

From a source checkout containing `pyproject.toml`:

```bash
python -m pip install .
```

For development installation and test prerequisites, see
[Contributing](./CONTRIBUTING.md). Package-index installation instructions apply
once a release is published; this documentation does not assert PyPI availability.

## Quick start

Pass an already loaded, supported model as `model_quantized`. The library does
not load or download a checkpoint:

```python
from llm_graph import DiagramConfig, LLMGraphError, constants, create_visualizer

visualizer = create_visualizer()
report = visualizer.visualize(model_quantized)
print(report.markdown())
```

The default produces one overview and four representative decoder variants with
their exact member indices. Representative block numbers, per-layer input slices,
and external KV references refer to the representative, not every member.

```python
report = visualizer.visualize(
    model_quantized,
    DiagramConfig(
        views=(constants.VIEW_OVERVIEW, constants.VIEW_DECODER),
        layer_index=19,
        direction=constants.DIRECTION_LEFT_RIGHT,
    ),
)
paths = report.write("model_diagrams")
```

Outputs include a combined Markdown report, JSON architecture/module inventory,
and separate diagram files. In a Mermaid-capable Markdown preview the fenced
diagrams render graphically. Other Markdown viewers may need Mermaid support
enabled; the library returns source, not a browser image.

```python
raw = visualizer.visualize(
    model_quantized,
    DiagramConfig(
        format=constants.BACKEND_MERMAID,
        views=(constants.VIEW_DECODER,),
        layer_index=0,
    ),
)
print(raw.artifacts[0].content)
```

`hierarchy` covers the full registered module tree, including collapsed
modalities' registered modules; it labels containment, not execution. It can be
large: select `views=("hierarchy",)` and deliberately raise `max_nodes` (for
example to 5000) if needed. The default node cap is 400 per rendered view;
inspection separately caps module registrations at 10000.
Mermaid hosts have their own limits (commonly 500 edges). Raising the library's
node limit does not raise the host limit: a full hierarchy may require a renderer
configured with larger `maxEdges`/`maxTextSize`, or use the smaller default views.
The parser test explicitly raises these limits for the full hierarchy.

## Errors

```python
try:
    report = visualizer.visualize(model_quantized)
except LLMGraphError as error:
    print(error.code, error.message)
    print(error.remediation)
    details = error.to_dict()
```

## Shared constants

`llm_graph.constants` is the dependency-free source of truth for all runtime
string values: issue codes, stages/severities, backend/view/format identifiers,
graph kinds and node IDs, property/serialization keys, model paths/version,
message/remediation text and rendering templates. Values use `Final[str]`;
callers can continue passing ordinary strings for backwards compatibility.
Constants avoid typos without changing public string values or serialized output.

```python
from llm_graph import constants

try:
    report = visualizer.visualize(model_quantized)
except LLMGraphError as error:
    if error.code == constants.ERROR_UNSUPPORTED_TRANSFORMERS_VERSION:
        print(error.remediation)
    else:
        raise
```

Add new vocabulary to `constants.py` and reference it at every runtime call site.
Do not duplicate a raw string in another module. Docstrings and `__all__` export
declarations are deliberately not extracted: they are Python documentation/export
metadata, not runtime model/report vocabulary. Tests enforce this distinction,
constant uniqueness, error-catalog coverage and valid template placeholders.
Invalid test inputs, mock targets and independent expected output may remain
literal in tests; supported vocabulary is shared with the implementation.

Runtime code never imports constants from documentation or tests.

## Error catalog details

See the [error directory](./ERRORS.md). The runtime immutable `CATALOG` contains
the fatal codes implemented by this release; the error directory also reserves
codes for future tracing/provider integrations. Diagnostics currently emitted:
`LLMG-I1001` (modality scope) and `LLMG-W1003` (config-only bit widths).
Non-PyTorch API objects currently receive `LLMG-E1001`, not provider-specific
classification. Error serialization excludes raw exception text, inputs and
tensor contents; chained local filesystem exceptions may contain full paths.

Writing refuses collisions unless `overwrite=True` is explicit. Files are
committed individually; directory-wide atomicity is not promised. A write error
includes committed artifact names. Do not share reports if model metadata itself
is confidential.

## Extension seams

`contracts.py` defines `ArchitectureAdapter` and `DiagramBackend` protocols.
Inject implementations through factories instead of changing a global registry:

```python
visualizer = create_visualizer(adapters=[custom_adapter], backends=[custom_backend])
```

The Gemma-only release uses graph-backed rendering requests; execution request
types in the broader plan will be introduced with an actual execution backend,
not speculative placeholders. `ViewBuilder` is independent of Torchview and
accepts graph data plus configuration. Verified status assumes the installed
vendor implementation has not been globally monkey-patched. Instance-level
forward overrides and unexpected text-module subclasses are rejected.

Compatibility promotion requires reviewing implementation changes and rerunning
architecture, quantizer, variant, safety and parser tests before extending the
tested version list. Matching names alone does not establish compatibility.

## Technical architecture

```text
Loaded model
  -> AdapterFactory -> Gemma4Adapter + ModuleInspector
  -> ArchitectureGraph (layers, modules, nodes, edges, tensor metadata)
  -> ViewBuilder (overview / variants / decoder / hierarchy)
  -> BackendFactory -> MermaidBackend
  -> VisualizationReport (Markdown, JSON, explicit artifact writing)
```

Adapters own model-specific interpretation; the inspector collects registered
metadata without accessing tensor values; the view builder selects bounded
representations; the renderer handles syntax rather than model semantics.
Factories use instance-owned injected registrations, not mutable global state.
The dependency-free constants module supplies shared identifiers and templates.

Public entry points are `create_visualizer`, `DiagramConfig`, `ModelVisualizer`,
`VisualizationReport`, `LLMGraphError`, and `constants`. A `py.typed` marker makes
the inline annotations discoverable by type checkers.

## Contributing

Contributions are welcome in documentation, tests, architecture adapters,
renderers, and compatibility validation. Start with a bounded issue or proposal
describing the use case, evidence, and acceptance checks before broad changes.
Follow [CONTRIBUTING.md](./CONTRIBUTING.md); new support must include verified
architecture evidence and tests, not just matching a model name.

## Release status

`llm-graph` is alpha software. Model coverage and public APIs may evolve;
compatibility and output changes should be reviewed before upgrading. Release
schedule, versioning, and compatibility commitments are documented in
[Release policy](./PUBLISHING.md).

## License

License terms have not yet been specified. This documentation does not grant
redistribution rights. Model repositories and dependencies retain their own
licenses and access conditions.

## Validation

Validation covers architecture topology, decoder grouping, quantization
metadata, deterministic output, coded failures, persistence, and Mermaid syntax.
Integration tests construct meta-device models with native quantization module
replacements, without loading checkpoint weights.

See [Testing](./CONTRIBUTING.md#testing) for commands and bundled fixture details.
The source repository includes packaging metadata, documentation, tests, and
development manifests; it does not depend on a parent project or model directory.
Mermaid parsing uses development-only Node
dependencies; no metadata is sent to an online rendering service. Loaded-model
acceptance remains separate from meta-model testing and memory benchmarking.
