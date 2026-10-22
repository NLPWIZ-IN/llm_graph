# Contributing to llm-graph

Development takes place in
[NLPWIZ-IN/llm_graph_view](https://github.com/NLPWIZ-IN/llm_graph_view).
The Python import remains `llm_graph`, and the distribution name remains `llm-graph`.

## Ways to contribute

Help make supported LLM architectures inspectable, accurately documented, and
reproducible without running inference. Priorities are correctness, provenance,
explicit limitations, deterministic output, and a small extensible architecture.
Read [README.md](./README.md), [DESIGN.md](./DESIGN.md), and
[ERRORS.md](./ERRORS.md) first. The design includes deferred features; confirm
current implementation scope in the README before proposing changes.

Useful contributions include:

- Correcting diagrams against verified model implementations.
- Portable small fixtures, regression tests, and compatibility validation.
- Improving API documentation, usage examples, and error remediation.
- Adding verified model adapters or new renderers behind existing contracts.
- Packaging and release automation with explicit credential boundaries.

## Reporting bugs

Report reproducible, non-sensitive bugs through the
[GitHub issue tracker](https://github.com/NLPWIZ-IN/llm_graph_view/issues).

Include a minimal reproducer, the expected and actual behavior, the `llm-graph`,
Python, PyTorch, and Transformers versions, and the model class and architecture
profile. For coded failures, include the error code and relevant sanitized
details. For incorrect diagrams, identify the affected view, layer index, and
the implementation evidence for the expected topology.

Do not include credentials, prompts, confidential model metadata, or checkpoint
weights. Reduce large examples to configuration-only or synthetic fixtures
where possible.

## Proposing changes

For initial questions, feedback, and feature discussions, join the
[llm_graph Slack community](https://join.slack.com/t/nlpwiz-in/shared_invite/zt-4e8mpwh4w-rtC~Dqbpuj7ymFawVEXD6g).
Keep actionable bug reports and agreed feature proposals in
[repository issues](https://github.com/NLPWIZ-IN/llm_graph_view/issues)
so decisions remain discoverable outside Slack. Community support is
best-effort; do not post confidential or security-sensitive information in
public channels.

For a new adapter, backend, dependency, or public API change, propose a bounded
design first. Include the motivating use case, supported model/implementation
versions, expected inputs/outputs, risks, and measurable acceptance criteria.
Discuss substantial changes with maintainers before implementation. Keep bug
fixes focused on a reproducible problem; separate unrelated refactoring from
behavior changes.

## Development setup

Use a Python 3.14 environment. From the directory containing `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
npm ci --ignore-scripts --no-audit --no-fund
python -m pyright --pythonpath "$(python -c 'import sys; print(sys.executable)')"
```

Node dependencies are development-only Mermaid syntax validation tools.
They are not needed by library users. Use an isolated development environment.

The repository uses a `src` layout: all runtime Python modules and the
`py.typed` marker live in `src/llm_graph/`. Setuptools discovers packages under
`src/`; tests, release scripts, documentation, and Node dependencies remain at
the project root and are not Python subpackages. Install the project before
running tests instead of relying on the working directory to resolve imports.
After changing the package layout, rerun the editable installation command above.

## Testing

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

The Gemma integration suite uses a bundled, independently authored JSON
architecture fixture. It constructs native modules on the PyTorch meta device
and applies native quantization replacements. No external configuration
directory, checkpoint weights, model downloads, or credentials are required.
The fixture represents the supported text dimensions and selected quantization
rules; it is not a copy of a checkpoint configuration and does not certify
checkpoint identity or multimodal tower behavior.

Run the independent constants tests with:

```bash
python -m unittest discover -s tests -p test_constants.py -v
```

Run focused regression tests during development, followed by all affected
integration and rendering checks before submission. Report missing fixtures and
skipped checks explicitly. Meta-model tests establish structural behavior, not
live tensor values or measured memory consumption.

### Continuous integration

The [Build and Test workflow](./.github/workflows/ci.yml) runs automatically for pull
requests targeting `main`: when opened, reopened, or updated with new commits
(`synchronize`). Pushes without an open pull request targeting `main`, direct
pushes to `main`, and pull requests targeting other branches do not trigger CI.
Manual runs are available for any branch containing the workflow through
Actions > Build and Test > Run workflow > Use workflow from. The workflow must
also exist on the default branch for manual dispatch to be available.
Superseded runs on the same pull request or manually selected branch are cancelled.

CI uses Ubuntu, Python 3.14, and Node.js 22. It installs the Mermaid validation
dependencies, checks Python dependency consistency and source types, builds the
source archive and wheel, validates distribution metadata, and runs the complete
test suite against the installed wheel rather than an editable checkout.
CI does not publish packages or require PyPI credentials; publication remains
in the separate release workflow. Using only pull-request events for automatic runs avoids
duplicate push and pull-request runs for the same update.
The publishing workflow also calls CI as a reusable workflow against the release
commit. Release building and publication require that validation to succeed.

## Implementation standards

- Preserve public behavior unless a change is intentional and documented.
- Keep inspection separate from architecture interpretation, view selection,
  rendering, and output writing. Prefer composition and dependency injection.
- Do not infer execution flow from module registration order.
- Do not run inference, dequantize tensors, move devices, or mutate the model in
  the non-executing inspection path.
- Use registered parameters/buffers for storage metadata. Avoid descriptors such
  as quantized embeddings' `weight`, which may dequantize a table.
- Put runtime string identifiers and templates in `src/llm_graph/constants.py` as
  `Final[str]` values. Reuse existing constants; preserve released code meanings.
- Keep types accurate; avoid blanket `Any` casts and suppressed diagnostics.
- Use standard coded exceptions for expected boundary failures. Preserve causes
  where useful; never catch broadly and return an empty successful report.
- Escape generated labels; do not introduce remote rendering or telemetry.
- Keep changes focused and update directly affected documentation.

## Adding an architecture adapter

1. Describe the exact model classes, framework versions, variant flags and
   quantization wrappers you will support.
2. Verify actual forward implementations, not just config or module names.
3. Document residual paths, normalization, position encoding, projection
   dimensions, attention types, shared state and optional branches.
4. Implement the existing adapter contract and inject registrations through the
   factory. Do not add architecture checks throughout the public facade.
5. Validate all properties used to label a graph; missing required facts must
   fail explicitly rather than use a plausible fallback.
6. Test tiny/meta models, malformed layouts, unsupported flags/versions, aliasing,
   quantization metadata, output determinism and absence of forward execution.
7. Provide a support-matrix entry and representative diagrams. Do not claim
   support for an entire family because one checkpoint works.

The current facade still adds Gemma-specific quantization metadata and the report
has Gemma-specific headings. Before registering a second model, move those
responsibilities behind adapter/report metadata contracts and add non-Gemma
tests. Registration alone is not sufficient for genuinely model-neutral output.

## Adding a renderer

Declare supported views and formats, preserve edge semantics, and validate
configuration before work begins. Test safe label handling, stable IDs, valid
syntax, limits, and media types. Do not reinterpret containment as execution or
silently substitute formats. An execution backend requires a separate approved
design for consent, input handling, mutation risks and resource limitations;
the current graph-rendering protocol alone does not establish tracing support.

## Contribution licensing and responsibility

Contributors retain ownership of any copyright they hold in their contributions.
By intentionally submitting material for inclusion in `llm_graph`, contributors
agree to license that material under the project's [MIT License](./LICENSE).

"llm_graph contributors" collectively refers to the respective copyright
holders. It does not imply that copyright is assigned to the repository owner
or to a separate legal entity.

### Contributor responsibilities

Contributors have primary responsibility for verifying the origin of, and their
authority to submit, the material they contribute. By submitting a contribution,
you confirm that:

1. **Submission rights:** You created the material or have sufficient permission
   to submit and license it under the project's MIT License.
2. **Third-party rights:** To the best of your knowledge, the contribution does
   not infringe copyright or other third-party rights, or breach confidentiality,
   employment, or contractual obligations.
3. **Disclosure and attribution:** You identify included third-party material,
   its source, applicable license, and required notices. Third-party material
   must remain clearly identified under its applicable terms.
4. **License compatibility:** You do not submit material whose terms prohibit
   its inclusion or distribution in this MIT-licensed project.
5. **No concealment:** You do not knowingly remove required notices, misrepresent
   authorship, or conceal restrictions affecting the contribution.
6. **Correction of concerns:** If you discover a rights or licensing concern
   after submission, you promptly notify the maintainers and reasonably cooperate
   in investigating and resolving it.

### Maintainer review

Acceptance or publication of a contribution does not transfer its copyright to
the repository owner or constitute independent legal clearance of the submitted
material.

Maintainers may request provenance or permission information and may reject,
remove, or replace contributions with unresolved rights or licensing concerns.

These contribution terms do not modify the MIT License, impose an
indemnification obligation, or determine liability under applicable law.
They do not eliminate any legal responsibilities of contributors, maintainers,
or distributors.

## Pull request checklist

- State the problem and scope, with before/after examples.
- Include regression tests that would fail without the change.
- Run focused tests, relevant integration tests, type checks and Mermaid parsing.
- Disclose skipped checks, missing fixtures and untested dependency versions.
- Keep output/code compatibility or document an intentional change.
- Update supported versions, error documentation and usage examples as needed.
- Exclude secrets, user prompts, weights, generated reports, environments,
  `node_modules`, caches and `*.egg-info`.
- Confirm that you have the right to contribute the code and any bundled material.
  Confirm compliance with the contribution licensing and responsibility terms
  above, including third-party disclosures and required notices.

Do not disclose security-sensitive reproductions in public issues. Use a private
maintainer reporting channel and provide only the information needed to
reproduce and assess the issue.
