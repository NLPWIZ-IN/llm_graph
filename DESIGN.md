# llm-graph architecture and extension design

Status: Gemma-only initial implementation; broader extension roadmap retained.

Source repository:
[NLPWIZ-IN/llm_graph_view](https://github.com/NLPWIZ-IN/llm_graph_view).
Architecture proposals and implementation discussions are tracked in
[repository issues](https://github.com/NLPWIZ-IN/llm_graph_view/issues).

## First-release scope override

The initial implementation targets
`google/gemma-4-E2B-it-qat-mobile-transformers`. See [README.md](./README.md)
for delivered APIs, installation, tests and limitations. The sections below
remain the broader extension design, not an implementation-completeness claim.

First release implements the Gemma adapter, Mermaid backend, graph schema,
module inspector, view builder, injected factories, typed coded errors, artifact
writer and installable package. To avoid speculative scaffolding, files are
organized by responsibility; adapters/backends can move into subpackages when
additional implementations justify that separation.

Llama/generic adapters, provider-client classifiers, Torchview, execution request
types and their resource limits are deferred. Unknown architectures and backend
names fail explicitly; no fallback suggests those features are implemented.
The runtime error catalog is a documented subset of the planned directory.
All implemented runtime string vocabulary and text templates are centralized in
`llm_graph/constants.py` as typed `Final[str]` values. Catalog definitions and
invocation sites reference the same code constants; consumers and tests reuse the
public `llm_graph.constants` module. AST-based tests forbid inline runtime strings
outside that module, excluding Python docstrings and export declarations.
Loaded-model acceptance remains separate from automated meta-model testing;
meta-model results do not establish live quantized storage behavior.

## 1. Requirements and agreed boundaries

Provide a Python library accepting an instantiated PyTorch model and producing
informative architecture diagrams and layer inventories. The input must not be
reloaded or downloaded.

Agreed initial scope:

- Verified adapters for installed Hugging Face Gemma 4 and Llama implementations.
- Generic PyTorch module hierarchy for other inspectable models, explicitly
  distinguished from a verified computation graph.
- Plugin contracts for additional architectures and visualization backends.
- Visualization backend resolved at runtime from configuration, including a custom
  Mermaid backend and an optional Torchview backend.
- Explicit unsupported-introspection errors for opaque Gemini/Claude API clients;
  no inferred proprietary layer diagrams or provider SDK dependencies.
- Review extension designs before adding new architecture or execution support.

"Any LLM" is an extension goal, not an accuracy guarantee. Family names and a
`generate()` method do not establish internal architecture compatibility.

## 2. Verified architecture profile

Architecture validation used Python 3.14, Transformers 5.19.0, implementation
source, and a configuration-only model constructed on the PyTorch meta device.
No checkpoint weights were required for structural validation.

- `_BaseModelWithGenerate` was not exported at runtime from the inspected
  `transformers.models.auto.auto_factory` module. Do not couple the public API to
  this private/editor-facing type. The relevant concrete model is a PyTorch
  `nn.Module`/Transformers `PreTrainedModel`.
- `AutoModelForCausalLM.from_config()` with the supported checkpoint config resolves to
  `Gemma4ForConditionalGeneration`, not the text-only causal class.
- Its text decoder is `model.language_model` under the model's `model` child.
  Vision/audio towers and modality embedders also exist.
- The text decoder contains 35 blocks: 28 sliding-attention and 7 full-attention.
- Hidden width is 1536. Sliding attention uses head width 256; full attention
  uses 512. Query projection width is therefore not simply the hidden width.
- Blocks 0-14 have independent K/V projections and MLP width 6144. Blocks 15-34
  reuse K/V states and have MLP width 12288; their attention modules omit K/V
  projection matrices.
- Inspected attention/KV/MLP properties partition the blocks into four groups:
  sliding/independent (12), full/independent (3), sliding/shared (16), and
  full/shared (4). Full-attention indices are 4, 9, 14, 19, 24, 29, 34
  (zero-based). Other properties must still be considered in variant signatures.
- Gemma 4 blocks include pre/post normalization, a gated per-layer input branch,
  and final layer scaling. Llama's inspected block has different norm placement.
- The supported Gemma config disables MoE. MoE-enabled Gemma variants are not silently
  treated as this verified dense variant.
- Quantization config is an object supporting `to_dict()`. Configured bit width,
  tensor storage dtype, logical matrix dimensions, and packed tensor shapes are
  different facts.
- Torchview's documented native graph representation is Graphviz; native
  Mermaid output is not an assumed backend capability.

Meta-model observations establish architecture, not the packed storage details
of a loaded quantized model. Those require inspection of that instance.

## 3. Accuracy and safety contract

Every report states its evidence level:

1. `verified_architecture`: supported adapter, validated concrete module layout,
   properties, variant flags, and supported Transformers implementation.
2. `module_hierarchy`: inspected containment only; no claimed execution order.
3. `observed_execution`: operations observed by a tracing backend for a particular
   input and execution configuration, not every possible model branch.

Every property records its source: module attribute, tensor metadata, model
config, verified implementation rule, or tracing observation. Missing information
is omitted with a diagnostic, not replaced with a plausible dimension.

Edges distinguish `data_flow`, `residual`, `contains`, `shared_state`, and
`weight_tie`. Hierarchy edges must never be rendered as execution arrows.

Default inspection:

- Does not run `forward()`/`generate()`, read tensor values, move devices, alter
  train/eval mode, register hooks, or mutate model/config.
- Does not use the network, automatically install dependencies, or execute remote
  model code.
- Supports meta tensors and quantized tensor metadata without dequantizing.
- Reads an allowlist of architectural properties, not arbitrary config contents,
  credentials, prompts, local filesystem paths, or parameter values.

## 4. Package boundaries and distribution

The public facade coordinates configuration, adapter selection, inspection,
view construction, rendering, and report persistence. Schema and protocol
definitions describe exchange types without depending on concrete adapters.
Architecture adapters own model semantics; rendering backends own output
syntax. Shared constants and the error catalog remain dependency-light.

Package the library as an installable distribution named `llm-graph`, exposing
the Python package `llm_graph`. The planned `pyproject.toml` uses setuptools with
an explicit mapping from the flat source directory to `llm_graph`, excluding
tests and development dependencies from Python package discovery. Declare
PyTorch/Transformers runtime dependencies and a `torchview` optional extra;
Graphviz's system executable is a separately documented prerequisite for SVG.
Dependency installation compatibility is distinct from verified adapter support.
Declare the initially tested Python environment and expand supported runtimes
only after testing; do not guess minimum dependency versions.

Core graph schema and Mermaid serialization use standard-library Python.
Install an editable source checkout in an isolated development environment:

```bash
python -m pip install -e ".[dev]"
```

The public import is `from llm_graph import ...`, independent of working directory.
No normal-use `sys.path` injection, CLI or plugin-discovery framework is needed.
`registry.py` holds factory implementations and immutable default registrations;
each factory owns its injected registrations, with no mutable global registry.

## 5. Classes, interfaces, and responsibilities

Use typed dataclasses and `typing.Protocol` at actual extension boundaries.
Avoid a class for every layer type and avoid deep inheritance.

| Type | Responsibility |
|---|---|
| `DiagramConfig` | Validated immutable backend, format, view, scope and size options |
| `TraceRequest` | Explicit example inputs and execution consent for tracing |
| `GraphNode`, `GraphEdge`, `Property`, `TensorInfo` | Backend-neutral, typed architecture facts |
| `ArchitectureGraph` | Nodes, typed edges, containment, evidence and provenance |
| `Diagnostic` | Stable code, severity, component path and actionable explanation |
| `IssueCode`, `IssueDefinition`, `IssueDetail` | Stable error/diagnostic identifiers, catalog definitions and safe invocation-specific details |
| `LLMGraphError` | Public base exception exposing a cataloged issue and JSON-safe serialization |
| `DiagramArtifact` | Mermaid text or SVG/DOT payload plus media type and view ID |
| `VisualizationReport` | Artifacts, evidence, diagnostics and optional architectural graph/inventory; execution-only reports do not fabricate these |
| `ModuleInspector` | Collects safe module/tensor metadata and shared identities |
| `ArchitectureAdapter` protocol | Matches and translates a supported model into an architecture graph |
| `Gemma4Adapter` | Verified text/multimodal Gemma wrappers and dense decoder variants |
| `LlamaAdapter` | Verified dense Llama decoder structure |
| `GenericModuleAdapter` | Containment graph only; never invents forward wiring |
| `AdapterFactory` | Resolves a registered adapter using explicit compatibility checks |
| `ViewSelection`, `GraphView` | Validated semantic view selection and bounded graph projection |
| `ViewBuilder` | Builds semantic graph views without access to the live model or tracing inputs |
| `DiagramBackend` protocol | Declares capabilities and produces requested artifacts |
| `MermaidBackend` | Renders the graph as deterministic escaped Mermaid text |
| `TorchviewBackend` | Adapts optional Torchview execution visualization to SVG/DOT artifacts |
| `BackendFactory` | Resolves backend by config and validates capabilities |
| `ModelVisualizer` | Orchestrates injected factories, inspector and view builder |
| `create_visualizer()` | Composition root wiring defaults or user-provided extensions |

### Core contracts (proposed signatures, not implemented code)

```python
ArchitectureAdapter.match(model: nn.Module) -> AdapterMatch
ArchitectureAdapter.inspect(
    model: nn.Module, inspector: ModuleInspector, scope: InspectionScope
) -> ArchitectureGraph

DiagramBackend.capabilities() -> BackendCapabilities
DiagramBackend.render(request: RenderRequest) -> tuple[DiagramArtifact, ...]

ViewBuilder.build(
    graph: ArchitectureGraph, selection: ViewSelection
) -> tuple[GraphView, ...]

ModelVisualizer.visualize(
    model: nn.Module,
    config: DiagramConfig | None = None,
    *,
    trace: TraceRequest | None = None,
) -> VisualizationReport
```

`AdapterMatch` distinguishes no match, compatible match, and recognized but
unsupported variant, with a reason. Resolution must not hide an unsupported
recognized architecture behind the generic fallback.

`RenderRequest` is a discriminated union: `GraphRenderRequest` contains graph
views and format; `ExecutionRenderRequest` contains the live model, explicit
`TraceRequest`, execution format and applicable limits. The facade constructs
requests after preflight; a backend declares accepted request kinds.
`ViewBuilder` receives only graph data and validated selection. It never imports
Transformers/Torchview, attaches a live model, or executes code on the model.
The persisted graph/report never holds the live model or input tensors.

The facade selects one of two pipelines:

```text
Semantic views:
model -> adapter -> ArchitectureGraph -> ViewBuilder -> graph backend

Observed execution:
model + explicit TraceRequest -> execution backend -> execution artifacts
```

Execution-only requests do not require a verified architecture adapter. They
still validate local input introspection support, backend capability and tracing
consent. Their report has `observed_execution` evidence and no architectural
graph/inventory unless separately requested through a semantic invocation.
Initial requests cannot mix semantic views and `execution`.

## 6. SOLID and pattern choices

- **Single responsibility:** inspection, architecture interpretation, view
  selection and serialization remain separate.
- **Open/closed:** register a new adapter/backend without editing orchestration.
- **Liskov substitution:** backends satisfy their declared capabilities; an
  unsupported format is an explicit error, not mislabeled output.
- **Interface segregation:** small adapter/backend protocols; no requirement for
  a Mermaid renderer to implement tracing.
- **Dependency inversion:** the facade receives factories/services; only the
  composition root knows concrete implementations.
- **Factory:** deterministic adapter/backend resolution.
- **Adapter:** converts framework-specific model layouts and Torchview outputs.
- **Strategy:** configured backends and view selection.
- **Composition:** graph containment represents nested blocks; use data rather
  than a Composite class hierarchy.

No singleton registry, service locator, automatic imports of third-party plugins,
decorator stack, or speculative Abstract Factory hierarchy is needed.

## 7. Inspection and adapter resolution

1. Validate input is an inspectable `nn.Module`; opaque API objects fail with
   `UnsupportedModelError`.
2. Resolve adapter using an explicit tested Transformers-version allowlist,
   concrete implementation identity, model/config type, variant flags and
   module-layout checks. Initially allow only 5.19.0 for verified adapters.
   `AdapterMatch` includes installed version, tested versions and rejection
   reasons. A recognized family outside the allowlist raises
   `UnsupportedArchitectureError` in architecture mode. Do not dispatch on
   display name alone or automatically extend compatibility to a version range.
3. Fail ambiguous matches with `AmbiguousAdapterError`. Explicit overrides must
   still pass compatibility checks.
4. Recognized but unverified revisions/variants fail in architecture mode.
   The caller can explicitly request hierarchy-only mode instead.
5. Unknown inspectable modules use the generic hierarchy adapter with a visible
   `LLMG-W1001` diagnostic. Strict architecture mode rejects them.
6. Extract metadata without reading weights. Traverse direct child registrations
   with an identity map so shared modules retain alias paths without recursion
   loops. Keep canonical IDs and alias relationships separate.
7. Validate graph integrity before view selection/rendering.

Start verified support against the inspected Transformers 5.19.0 implementations.
Version/layout checks are necessary, not sufficient: remote subclasses with
overridden forward behavior, patched implementations, unsupported flags, and
unrecognized quantized wrappers cannot acquire verified status through matching
names alone. Additional revisions require fixtures and explicit compatibility
updates. Do not claim universal semantic equivalence from runtime introspection.

### Compatibility maintenance

Each adapter publishes an exact tested-version list, supported concrete classes,
variant flags and quantized-wrapper coverage. Expanding that list requires:

1. Review candidate-version changes to supported forward paths and quantizers.
2. Run tiny/meta architecture fixtures and version-specific negative tests.
3. Verify residual/norm wiring, per-layer dimensions, KV sharing, variant
   signatures and quantized wrapper metadata.
4. Update adapter compatibility metadata and the documented support matrix.
5. Add the version only when these checks pass; never widen a version range based
   on matching names/layout alone.

New versions remain unverified until this procedure is completed. Explicit
hierarchy mode remains available without weakening verified-architecture claims.
Automated tests cover accepted and rejected versions; adding a tested version
must not require changing facade or backend code.

### Layer properties

Common: canonical module path, concrete class, semantic kind, logical input/output
width, normalization epsilon, activation, bias, parameter/buffer counts, dtype,
device, stored tensor shape, quantization metadata and shared-identity references.

Attention: query/KV head counts, actual head width, projection widths, causal mode,
window size, positional mechanism, Q/K/V normalization, KV producer/consumer links.

MLP: actual intermediate width, gate/up/down projections and gating activation.
Normalization: exact placement, normalized dimension, epsilon and learned scale.
Embeddings/head: vocabulary size, hidden width and verified weight ties.

Per-module quantization precedence must follow the installed Gemma quantizer's
matching rules, not an assumed "first regex wins" rule. Configured bits must be
labeled as configured, unless the concrete packed implementation verifies them.
Do not equate `numel * dtype_size` with true model/device memory usage. Separate
logical parameter counts, stored tensor counts, and alias-aware storage metrics;
omit metrics that cannot be established safely.

## 8. Graph construction and views

Graph validity: unique IDs, existing edge endpoints, acyclic containment, explicit
input/output boundaries, and no shared-state edge masquerading as token flow.

Supported Mermaid views:

- `overview`: embeddings, ordered decoder groups, final norm and output head;
  multimodal towers/projectors may be shown collapsed with scope labels.
- `decoder`: one explicitly selected block, including functional operations such
  as residual addition, normalization, gate multiplication and scaling.
- `variants`: one diagram per structurally distinct decoder signature, with exact
  member indices. Compute signatures from inspected properties, not family-wide
  assumptions. Include dimensions, KV producer/consumer role and branch layout.
  Concrete external producer indices remain instance metadata and do not alone
  prevent grouping equivalent blocks.
- `hierarchy`: registered module containment only, suitable for unknown models.

When `views` is omitted, verified Mermaid adapters default to overview plus
decoder variants; the generic adapter defaults to hierarchy and a visible
coverage diagnostic. Torchview defaults to `execution`. Explicit
view requests are never substituted. Full per-block expansion is opt-in.
Ordered overview grouping uses contiguous compatible runs and never reorders
alternating attention types. Cross-layer KV links retain producer indices even
when intervening blocks are collapsed.

For Gemma 4, distinguish sliding/full attention and KV-producing/KV-consuming
blocks, doubled MLP widths and per-layer inputs. Verify producer mapping and
per-layer embedding preparation against installed forward code during
implementation. Never draw missing K/V matrices in shared-KV layers.

Multimodal internal detail depends on backend capabilities and verified coverage.
An expanded scope not supported by the selected adapter/backend raises
`UnsupportedViewError`; it does not silently collapse requested content.
Validate all requested views first. An unsupported view aborts the entire call
with no returned partial report and no file writes; initial APIs do not offer
best-effort partial rendering.

Resource limits have separate meanings:

| Limit | Pipeline | Enforcement |
|---|---|---|
| `max_modules` | Semantic inspection | Bound visited module registrations/aliases before full graph completion; fail with `LLMG-E3001` |
| `max_nodes` | Semantic view | Bound selected rendered nodes per view; fail with `LLMG-E4005` before rendering |
| `max_trace_nodes` | Execution | Count observed graph nodes after trace collection and before DOT/SVG serialization; fail with `LLMG-E4007` |

Never silently truncate a graph. `max_nodes` is not an execution-memory limit.
Initial Torchview integration makes no collection-time node, hard timeout,
CPU/GPU memory or process-isolation guarantee. A post-trace node limit cannot
prevent tracing-time resource exhaustion. Document backend-specific counting
and use tiny inputs/prepared models for tests. Requests for unsupported hard
execution budgets fail preflight with `LLMG-E2001` rather than implying protection.
Adding collection-time enforcement requires demonstrated backend support and
separate tests. No claim of constant memory for arbitrary models is made.

## 9. Runtime backend and format configuration

Architecture coverage and rendering backend are separate dimensions. Selecting
Torchview does not supply an unimplemented semantic adapter or prove every path.

`ViewKind` has the canonical identifiers below. Meaning does not change with
backend; unsupported pairs raise `LLMG-E2004`, never coerce to another view.

| View | Meaning | Mermaid | Torchview |
|---|---|---|---|
| `overview` | Verified high-level architectural flow | Supported with verified adapter | Unsupported initially |
| `decoder` | Verified selected decoder block | Supported with verified adapter | Unsupported initially |
| `variants` | Verified structural decoder variants | Supported with verified adapter | Unsupported |
| `hierarchy` | Registered module containment only | Supported | Unsupported initially |
| `execution` | Operations observed for explicit inputs | Unsupported initially | Supported |

| Backend | Format identifiers | Execution | Request kind |
|---|---|---|---|
| `mermaid` | `mermaid`, `markdown` | No | Graph |
| `torchview` | `svg`, `dot` | Yes, explicit inputs/consent | Execution |

`DiagramConfig.views=None` selects the documented defaults in section 8;
an explicitly empty tuple is invalid (`LLMG-E2001`). Unknown view identifiers
are invalid configuration; known unsupported combinations are `LLMG-E2004`.
Mixed semantic/execution selections are unsupported. Resolve omitted format to
`markdown` for Mermaid or `svg` for Torchview; explicit formats are never replaced.
Semantic requests reject a supplied `TraceRequest` as unused configuration.
Execution requests require `TraceRequest` and consent (`LLMG-E4002`), then
compatible inputs/setup (`LLMG-E4003`).

For example, `backend="torchview", views=("overview",)` raises `LLMG-E2004`
before inference. A valid Torchview request uses `views=("execution",)` and
`format="svg"` or `"dot"`. `ViewBuilder` is not involved in that request.

Torchview decoder-submodule tracing requires compatible caller-supplied kwargs;
it cannot assume token IDs satisfy a decoder's hidden-state/position inputs.
Do not promise semantic variant/overview collapsing through Torchview unless its
integration can implement those views faithfully. Unsupported combinations fail.

Resolve backend and validate format/view/dependencies before tracing. No silent
fallback from Torchview to Mermaid or from Mermaid output to an SVG.
Import Torchview lazily; missing Python packages or Graphviz executables yield
`MissingBackendDependencyError` with setup guidance. Do not install automatically.

Tracing executes user model code and may allocate memory, update state, or have
custom side effects. It is opt-in and never part of the safe inspection default.
Require the caller to supply a prepared evaluation model and example inputs on
the intended device. Do not call `.to()` or silently switch the original model
to meta. Document Torchview's behavior after integration testing and reject
options that move or modify the caller's model. Arbitrary forward side effects
cannot be universally rolled back; do not promise transactional tracing.
Wrap backend exceptions with context while preserving their original cause.

## 10. Rendering and output

- Stable node IDs derived from canonical paths, not memory addresses.
- Escape label text and property values; whitelist Mermaid direction/style tokens.
  Model-provided strings cannot inject graph syntax, links or HTML.
- Different line styles and a legend distinguish containment, data, residual and
  shared-state edges. Residual additions are explicit operation nodes.
- Show key properties on nodes; put exhaustive metadata in the inventory.
- Return artifacts in memory; no implicit file writes.
- Provide an explicit report writer with an output directory and overwrite flag.
  Use generated safe filenames and relative links in Markdown; refuse overwrite
  by default. SVG/DOT names and extensions must match artifact media types.
- JSON report serialization uses plain data and a schema version; no pickle.
- Do not send model metadata or diagrams to remote rendering services.

Illustrative intended usage (not executable until implementation):

```python
from llm_graph import DiagramConfig, create_visualizer

visualizer = create_visualizer()
report = visualizer.visualize(
    model_quantized,
    DiagramConfig(backend="mermaid", format="markdown", views=("overview", "variants")),
)
```

Install into the application environment as described in section 4. Test
this import from an unrelated working directory, not only the package root.
Constructor injection also allows a caller to provide factories containing
custom adapters/backends instead of global registry mutation.

## 11. Explicit failures and diagnostics

The proposed [error directory](./ERRORS.md) defines stable codes, exception
categories, message templates, recovery guidance and non-fatal diagnostics.
It is part of the public API contract, not just developer notes.

`errors.py` owns a typed immutable catalog and public exceptions; it has no
dependency on model adapters or renderers. `IssueCode` is a string enum,
`IssueDefinition` contains the canonical template/severity/remediation, and
`IssueDetail` holds safe invocation-specific fields. `LLMGraphError` exposes
`code`, `message`, `stage`, `component_path`, `details`, `remediation`, and
`to_dict()`. Specialized exceptions preserve convenient category-based catching;
one class per code is unnecessary.

Default invocation raises a typed exception for fatal errors. `str(error)`
includes its stable code and actionable message, so application tracebacks report
the failure without a separate logging service. Successful reports carry typed
non-fatal diagnostics using the same catalog conventions. No success-shaped
error report, automatic retry, silent fallback, or duplicate automatic logging.
Optional caller logging uses structured serialization; it is not a library
dependency. Raw backend exceptions are chained, not serialized into reports.

All requested views are prevalidated before rendering, and no partial report is
returned on failure. Required shape/topology inconsistencies are fatal; optional
missing properties can produce diagnostics. Error codes never change meaning
and are never reused; tests validate unique codes, template placeholders, safe
serialization and representative failure paths.

## 12. Implementation phases and acceptance tests

1. **Foundation:** schema, contracts, configuration, factories, inspector, facade.
   Add the installable manifest and stable public exports. Implement the error
   catalog/base exception first. Test invalid inputs/config,
   ambiguity, alias/cycle handling, graph validity and stable failure codes.
2. **Mermaid + generic hierarchy:** deterministic rendering, escaping, inventory,
   output writer, limits and explicit hierarchy labeling.
3. **Llama adapter:** tiny random/meta local models; verify norm/residual wiring,
   Q/K/V/O widths, GQA, MLP activation and tied/untied weights.
4. **Gemma 4 adapter:** text and multimodal wrappers; verify sliding/full attention,
   KV producer-consumer links, widths 6144/12288 for the supported config, per-layer
   input branch, Q/K/V norms, scaling and absent K/V projection nodes.
5. **Optional Torchview:** missing-dependency behavior without installation;
   optional integration test with tiny prepared inputs, valid SVG/DOT output,
   explicit format/view rejection and observed state/device behavior.
6. **Documentation and examples:** public usage and plugin examples; do not
   require application-specific setup or automatically run inference.

Validation must include:

- Configuration-only meta model: exactly 35 decoder blocks, 28 sliding and 7 full,
  independent K/V in the first 15 and shared-KV in the remaining 20.
- Attention/KV/MLP grouping must reproduce the observed four groups with counts
  12, 3, 16 and 4 and their exact block indices. Test the complete structural
  signature separately so additional branch/property differences are not lost.
- Inspect loaded quantized modules in a separate acceptance environment; no
  additional multi-gigabyte checkpoint load for automated tests.
- Tiny quantized metadata fixtures distinguish logical vs stored dimensions and
  verify installed quantizer override rules; unsupported wrappers fail clearly.
- No forward call, device move, weight read or model mutation in Mermaid mode.
- Unknown modules never emit verified residual/data-flow edges.
- Deterministic output, repeated layers not merged across distinct signatures,
  and every alias/shared edge references an existing canonical node.
- Parse generated Mermaid with a locally available Mermaid parser/CLI during
  implementation. If absent, obtain/install the minimal validation tooling before
  declaring renderer syntax verified; snapshot comparison alone is insufficient.
  This is development-only tooling, installed only after declaring its dependency
  or encountering a missing-tool validation failure, never library runtime setup.
- Reject untested Transformers versions, ambiguous adapters, unsupported views in
  multi-view requests, and recognized-but-unsupported variants. Test injected
  registrations do not leak between visualizer instances.
- Render stress test with many layers against explicit node/traversal limits.
- Type checking and focused tests using existing tooling when available.
- Test `ViewBuilder` using synthetic graphs without importing model frameworks
  or execution backends. Test every backend/view/format capability combination,
  omitted defaults, mixed-view rejection and no inference on preflight failure.
- Test editable installation/import from an unrelated working directory.
- Test semantic module/view limits separately from post-trace node limits.
  An oversized tiny trace raises `LLMG-E4007` before artifact serialization;
  documentation must not describe that as collection-time memory protection.
- Error contract: opaque clients produce `LLMG-E1002`; invalid inputs, unknown
  backends, incompatible formats, unsupported versions, missing dependencies,
  tracing refusal, inspection limits and write failures have their documented
  codes. Exceptions expose JSON-safe allowlisted details without model values,
  prompts, credentials, full config dumps or raw backend messages.
- Catalog integrity: no duplicate/reused codes, exact message placeholders,
  deterministic formatting, category inheritance, diagnostic severities and
  catalog/documentation parity.

### End-to-end acceptance gate

After implementation, run this consolidated user journey:

1. Install the package into an isolated environment and import `llm_graph` from
   an unrelated working directory using only documented setup steps.
2. Pass the existing live `model_quantized` to Mermaid with `overview`/`variants`;
   do not reload the checkpoint. Confirm no forward/generate call, device/dtype
   move or model/config mutation. Use call spies that delegate to the original
   method when needed, not an overridden forward implementation that would
   invalidate architecture-compatibility checks.
3. Verify all 35 blocks, the four attention/KV/MLP groups with their exact indices,
   actual dimensions, absent K/V matrices and producer/consumer references.
4. Where bit widths are config-only, require `LLMG-W1003`; do not assert measured
   storage or require a warning when storage has genuinely been verified.
5. Parse every Mermaid artifact locally; write the report and relative links.
   Repeat writing with overwrite disabled (`LLMG-E5001`), then explicit overwrite.
6. Pass a recognized opaque-client fixture; assert `LLMG-E1002` and the complete
   documented serialized issue shape without network activity.
7. In an optional-backend environment, trace a tiny prepared model with explicit
   inputs/consent; validate SVG/DOT and `observed_execution` reporting. Confirm
   `torchview` plus `overview` fails preflight with `LLMG-E2004` and the post-trace
   limit produces `LLMG-E4007` without serializing oversized artifacts.

Tiny fixtures remain the automated baseline. If no loaded supported model is
available, record the live-quantized acceptance step as pending rather than
reloading weights or claiming full end-to-end verification.

## 13. Review gate

Independent review should challenge architecture fidelity, public type choices,
backend capability boundaries, generic fallback honesty, quantized metadata,
state safety, plugin resolution, and whether the design is unnecessarily complex.

An independent read-only design review recommended approval with non-blocking
findings. The following revisions address every reported finding:

| Finding | Disposition |
|---|---|
| Version compatibility was not explicit in matching inputs | Added exact tested-version allowlist, match metadata, failure behavior and tests in sections 7/12 |
| Joint attention/KV variant grouping needed acceptance criteria | Verified all blocks on the meta model; added four observed groups and computed-signature requirements in sections 2/8/12 |
| Registry naming could imply mutable global state | Clarified instance-owned registrations and immutable defaults in section 4 |
| Multi-view failure behavior was unspecified | Defined validation-first, all-or-error semantics in section 8 |
| Runtime vs test dependency installation was ambiguous | Distinguished development-only parser tooling from runtime behavior in section 12 |

Also clarified capability-aware defaults when views are omitted, avoiding a
generic hierarchy adapter receiving unsupported default decoder views.

### Subsequent independent critique and revision

A second independent critique included both this plan and the error catalog.
It assessed the design at 69.8/100 and identified one blocking contract gap plus
five additional findings. This revision addresses those findings as follows:

| Finding | Revision |
|---|---|
| Backend/view vocabulary mismatch | Canonical view matrix, explicit defaults/rejections and separate semantic/execution paths in sections 5/9 |
| Missing `ViewBuilder` contract | Added pure graph-view signature and facade ownership of discriminated render requests in section 5 |
| Diagnostic vocabulary drift | Replaced informal diagnostic name with cataloged `LLMG-W1001` in section 7 |
| Deferred packaging | Selected editable setuptools package, manifest and cwd-independent `llm_graph` import in sections 4/10/12 |
| Version allowlist maintenance | Added explicit candidate-version verification and support-matrix procedure in section 7 |
| Torchview limit ambiguity | Separated module/view/post-trace limits, added `LLMG-E4007`, and documented absence of hard execution budgets in sections 8/12 and the error directory |

Added a consolidated model-to-artifact acceptance gate in section 12.
These are documented resolutions, not implementation verification or a new
independent approval of the revised text.

These reviews assess the design rather than certify implementation behavior.
See the README for delivered capabilities and the contribution guide for
validation requirements. Deferred backend acceptance tests apply only when
those backends are implemented.
