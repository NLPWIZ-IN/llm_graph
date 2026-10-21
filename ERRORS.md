# Error and diagnostic directory: proposed public contract

Status: catalog for the broader design, partially implemented in the Gemma-only release.
See [DESIGN.md](./DESIGN.md) for architecture and invocation behavior.

Report reproducible, non-sensitive error-handling problems in the
[NLPWIZ-IN/llm_graph_view issue tracker](https://github.com/NLPWIZ-IN/llm_graph_view/issues),
following the [bug-report guidance](./CONTRIBUTING.md#reporting-bugs).

Implemented fatal codes are listed in the immutable `llm_graph.errors.CATALOG`:
`E1001`, `E1101`-`E1105`, `E2001`-`E2004`, `E3001`-`E3003`, `E4005`,
and `E5001`-`E5003` (each with prefix `LLMG-`). Codes for opaque-client
classification, tracing, plugin-specific registration errors and other deferred
paths remain planned, not implemented. Current non-fatal diagnostics are
`LLMG-I1001` and `LLMG-W1003`. See [README.md](./README.md) for release scope.

The implemented code strings and their canonical templates/remediations are
declared once in `llm_graph.constants`. Use symbolic constants such as
`constants.ERROR_INVALID_MODEL_INPUT`, `constants.ERROR_UNSUPPORTED_ARCHITECTURE`
and `constants.WARNING_CONFIGURED_QUANTIZATION`, rather than repeating code
strings in Python. The catalog and exception call sites share these definitions;
tests verify their coverage, uniqueness and unchanged wire-format string values.

## 1. Conventions

- Errors use `LLMG-E####`; warnings use `LLMG-W####`; informational diagnostics
  use `LLMG-I####`. Codes are stable strings, independent of exception class names.
- Codes identify causes, not HTTP statuses. No provider API call is required.
- Once released, never reuse a code or change its meaning. Add a new code for a
  different condition; retain deprecated catalog entries.
- Messages are human-readable; callers branch on `code`, never parsed messages.
- Templates use named placeholders containing validated, bounded scalar values.
  Never interpolate a model object's representation or arbitrary config dump.
- Fatal conditions raise exceptions. Non-fatal diagnostics are returned only
  with a valid visualization report.
- No condition automatically retries or changes the chosen backend/scope.
- These are the planned initial codes, not a guarantee that arbitrary Python or
  operating-system failures can all be exhaustively classified.

## 2. Structured invocation contract

`LLMGraphError` is the public base exception. Category subclasses share an
immutable `IssueDetail` containing:

| Field | Meaning |
|---|---|
| `code` | Stable string enum value |
| `severity` | `error`, `warning`, or `info` |
| `message` | Formatted canonical message |
| `stage` | `validation`, `resolution`, `inspection`, `view`, `render`, or `write` |
| `component_path` | Optional canonical module path, not an absolute filesystem path |
| `details` | Typed, JSON-safe allowlisted context; strings, numbers, booleans, nulls, lists and maps only |
| `remediation` | Actionable recovery instruction |

`str(error)` is `[CODE] MESSAGE`. `error.to_dict()` serializes the fields above
and an `issue_schema_version` initially set to `1`. Exception type, raw traceback,
model objects, tensors and chained exceptions are not serialized.

The original cause is available through normal Python exception chaining for
local debugging. Such tracebacks may contain sensitive third-party information;
applications must not forward them externally without their own redaction.

Example serialized failure:

```json
{
  "issue_schema_version": 1,
  "code": "LLMG-E1002",
  "severity": "error",
  "message": "Layer introspection is unavailable for opaque model object 'RemoteModelClient'.",
  "stage": "validation",
  "component_path": null,
  "details": {"input_type": "RemoteModelClient"},
  "remediation": "Pass a locally inspectable PyTorch model; internal layers cannot be recovered from an API client."
}
```

## 3. Fatal error directory

### Input and architecture: 1xxx

| Code | Exception category | Canonical message template | Recovery |
|---|---|---|---|
| `LLMG-E1001` | `InvalidModelInputError` | Expected an inspectable PyTorch model; received '{input_type}'. | Pass an instantiated local model, not a name, path, null, or response object. |
| `LLMG-E1002` | `UnsupportedModelError` | Layer introspection is unavailable for opaque model object '{input_type}'. | Pass a local inspectable model; do not infer provider internals. |
| `LLMG-E1101` | `UnsupportedArchitectureError` | Architecture '{model_type}' has no verified adapter for the requested mode. | Register a compatible adapter or explicitly request hierarchy-only mode. |
| `LLMG-E1102` | `UnsupportedArchitectureError` | Adapter '{adapter}' does not verify Transformers version '{installed_version}'. | Use a tested version or explicitly request hierarchy-only mode. |
| `LLMG-E1103` | `UnsupportedArchitectureError` | Adapter '{adapter}' does not support variant '{variant}'. | Add verified variant support or explicitly choose hierarchy-only mode. |
| `LLMG-E1104` | `AmbiguousAdapterError` | Multiple compatible adapters match architecture '{model_type}'. | Select a compatible adapter explicitly or remove conflicting registrations. |
| `LLMG-E1105` | `UnsupportedArchitectureError` | Architecture layout differs from adapter '{adapter}' at '{component_path}'. | Inspect the implementation and update/test the adapter; do not guess wiring. |
| `LLMG-E1106` | `InspectionError` | Required property '{property}' cannot be established at '{component_path}'. | Supply a supported implementation or fix the adapter's property extraction. |

Opaque-client classification uses a registered, side-effect-free input classifier
or a recognized client type. Arbitrary non-module objects are `E1001`, not
automatically branded proprietary clients. No imports of provider SDKs or network
probes are needed. If an opaque wrapper is itself an `nn.Module` but exposes no
inspectable model internals, its adapter must reject verified introspection
rather than accept an empty module graph.

### Configuration and backend resolution: 2xxx

| Code | Exception category | Canonical message template | Recovery |
|---|---|---|---|
| `LLMG-E2001` | `InvalidConfigurationError` | Invalid value for configuration field '{field}': {reason}. | Correct the field according to the documented configuration schema. |
| `LLMG-E2002` | `InvalidConfigurationError` | Visualization backend '{backend}' is not registered. | Select an available backend or inject its registration. |
| `LLMG-E2003` | `InvalidConfigurationError` | Backend '{backend}' does not support format '{format}'. | Choose a supported format; Torchview output is not native Mermaid. |
| `LLMG-E2004` | `UnsupportedViewError` | Backend '{backend}' with adapter '{adapter}' does not support view '{view}' for scope '{scope}'. | Choose a supported view/scope or add verified support. |
| `LLMG-E2005` | `InvalidConfigurationError` | Invalid or conflicting {registry_kind} registration '{registration_id}'. | Correct the factory registrations before invoking visualization. |
| `LLMG-E2006` | `UnsupportedArchitectureError` | Explicit adapter '{adapter}' is incompatible with architecture '{model_type}'. | Choose a compatible adapter; explicit selection does not bypass validation. |

### Inspection and graph construction: 3xxx

| Code | Exception category | Canonical message template | Recovery |
|---|---|---|---|
| `LLMG-E3001` | `InspectionLimitError` | Inspection exceeds configured {limit_name} limit {limit}. | Increase the deliberate safety limit or narrow inspection scope. |
| `LLMG-E3002` | `InvalidGraphError` | Graph invariant '{invariant}' failed at '{component_path}'. | Correct the adapter/plugin; no invalid graph is rendered. |
| `LLMG-E3003` | `InspectionError` | Required property '{property}' conflicts between module and config at '{component_path}'. | Resolve the incompatible model/config or implement supported precedence. |
| `LLMG-E3004` | `InspectionError` | Module metadata inspection failed at '{component_path}'. | Inspect the chained local cause and fix the unsupported module accessor. |

### Rendering and tracing: 4xxx

| Code | Exception category | Canonical message template | Recovery |
|---|---|---|---|
| `LLMG-E4001` | `MissingBackendDependencyError` | Backend '{backend}' requires missing dependency '{dependency}'. | Install the documented optional package or executable yourself. |
| `LLMG-E4002` | `TraceConfigurationError` | Backend '{backend}' requires explicit consent to execute the model. | Supply a deliberate tracing request or choose non-executing Mermaid inspection. |
| `LLMG-E4003` | `TraceConfigurationError` | Tracing input or execution setup is invalid: {reason}. | Supply compatible example inputs, prepared model mode and supported device setup. |
| `LLMG-E4004` | `BackendExecutionError` | Backend '{backend}' failed while producing view '{view}'. | Inspect the local chained cause; reduce trace scope or correct the backend setup. |
| `LLMG-E4005` | `RenderLimitError` | View '{view}' requires {node_count} nodes, exceeding limit {limit}. | Select a collapsed/block view or explicitly raise the rendering limit. |
| `LLMG-E4006` | `TraceConfigurationError` | Requested tracing option '{option}' would violate the model-preservation policy. | Prepare a separate model explicitly or remove the mutating option. |
| `LLMG-E4007` | `RenderLimitError` | Observed execution graph has {node_count} nodes, exceeding post-trace limit {limit}. | Reduce trace scope/inputs or deliberately raise the post-trace output limit; this limit does not protect collection-time memory. |

`max_modules` bounds semantic inspection (`LLMG-E3001`); `max_nodes` bounds
semantic view selection (`LLMG-E4005`); `max_trace_nodes` bounds the collected
execution graph before serialization (`LLMG-E4007`). Define the trace count as
observed computation nodes in the backend graph, excluding presentation-only
clusters/legend nodes; verify the extraction against the chosen Torchview version.
If the integration cannot reliably count those nodes, it must reject the requested
limit with `LLMG-E2001`, not pretend enforcement. None of these limits promises
a hard tracing timeout or CPU/GPU memory bound.

Canonical views are `overview`, `decoder`, `variants`, `hierarchy`, and `execution`.
Torchview initially supports only `execution`; Mermaid supports only the first
four, subject to adapter coverage. Known unsupported backend/view combinations,
including mixed semantic/execution requests, raise `LLMG-E2004` before execution.
For execution-only errors its `{adapter}` context is the documented sentinel
`not_applicable`, not a fictitious verified adapter. Unknown view IDs, empty
explicit selections, unused trace requests on semantic calls, and unsupported
hard execution-budget settings raise `LLMG-E2001`. Omitted trace requests or
missing consent on execution calls raise `LLMG-E4002`.

Backend execution faults include unsupported tracing operations and execution
failures; do not classify every backend exception as a missing dependency.
Preflight missing dependencies are detected narrowly. Do not catch `BaseException`
or mask interrupts/exits, and do not treat resource-exhaustion failures as a
reason to retry automatically.

### Explicit artifact writing: 5xxx

| Code | Exception category | Canonical message template | Recovery |
|---|---|---|---|
| `LLMG-E5001` | `ArtifactWriteError` | Artifact '{artifact_name}' already exists and overwrite is disabled. | Select another directory or explicitly enable overwrite. |
| `LLMG-E5002` | `ArtifactWriteError` | Cannot write artifact '{artifact_name}'. | Check the destination, permissions and available disk space. |
| `LLMG-E5003` | `InvalidConfigurationError` | Artifact name '{artifact_name}' is not safe for the output directory. | Use a validated relative filename without traversal or absolute paths. |

Multi-artifact writing preflights names/collisions and uses temporary files plus
per-file atomic replacement where available. A write failure never returns a
success result. Whole-directory transactional atomicity is not promised; a
failure includes bounded safe names of artifacts already committed so the caller
can reconcile partial output. Temporary files created by this invocation are
cleaned up; pre-existing user files are never deleted for rollback.

## 4. Non-fatal diagnostic directory

| Code | Severity | Canonical message template | Meaning/action |
|---|---|---|---|
| `LLMG-W1001` | warning | Architecture '{model_type}' is shown as module hierarchy only. | Execution wiring is unverified; register a verified adapter if needed. |
| `LLMG-W1002` | warning | Optional property '{property}' is unavailable at '{component_path}'. | Omitted property is unknown; no inferred substitute. |
| `LLMG-W1003` | warning | Quantization width at '{component_path}' is configured, not storage-verified. | Do not interpret configured bits as measured memory usage. |
| `LLMG-W1004` | warning | Optional property '{property}' differs between module and config at '{component_path}'. | Report both sources; required conflicts instead raise `E3003`. |
| `LLMG-I1001` | info | Modality component '{component_path}' is collapsed under the requested scope. | Intentional, explicitly supported scope boundary. |
| `LLMG-I1002` | info | Backend '{backend}' shows only execution observed for the supplied inputs. | Unexecuted branches are not verified by this trace. |

## 5. Invocation behavior and ownership

1. Validate input/config, resolve adapter/backend, and prevalidate requested
   capabilities/dependencies. Return the first deterministic fatal issue; do not
   aggregate unrelated runtime failures or execute a trace to discover setup errors.
2. Classify expected failures where their cause is known, using the narrowest
   category. Preserve causes for backend, inspection and filesystem failures.
3. Raise public typed exceptions for documented boundary failures. Unexpected
   programming errors remain visible with their traceback; never return an empty
   report or relabel an unknown defect as unsupported architecture.
4. Return non-fatal diagnostics only with a valid report. They do not invoke
   Python warning filters or duplicate logging automatically.
5. Callers can catch `LLMGraphError` and display its message or serialize its
   allowlisted fields. No provider calls, HTTP mapping or logging framework is
   built into the library.

Classify in this order within validation/resolution: malformed input/config,
input introspection support, backend registration and format, adapter compatibility,
joint view/scope capability, dependencies and trace setup. Do not construct the
full graph or execute the model until these checks pass.
Adapter compatibility applies only to semantic requests. Execution-only requests
skip verified adapter resolution, use the backend's explicit `execution`
capability, and do not fabricate an architecture graph/inventory. Semantic view
selection and serialization remain separate from execution request handling.

Use `LLMG-W1001` consistently for hierarchy-only coverage; it is the cataloged
diagnostic, not a second uncataloged `UNVERIFIED_ARCHITECTURE` issue.

The immutable runtime catalog is authoritative. Tests compare its entries to
this directory, verify exact named template placeholders and unique codes, and
exercise each code through a focused fixture or constructor/invariant test.
Extensions may raise the public category exceptions using core codes only when
the documented meaning matches; otherwise use plugin-owned namespaced codes such
as `acme.graph:E1001`. They must not modify or reuse `LLMG-*` definitions.

All details/messages have size limits and safe scalar formatting. No raw
`repr(model)`, arbitrary backend exception text, full local paths, environment
dumps, input prompts, token values or tensor contents enter serialized issues.
Explicitly documented module paths and bounded class/adapter identifiers are
permitted. Remediation remains user-readable, while automation uses stable codes.
