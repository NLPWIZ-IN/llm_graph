# Release policy

This document describes the release schedule, versioning strategy, and backward
compatibility commitments for `llm-graph`. Release management, approval,
publication, and maintenance of this policy are the responsibility of the
repository owner.

## Release schedule

Starting in **November 2026**, regular releases are scheduled for the **third
Sunday of each month**. The first scheduled release date is **15 November 2026**.

The schedule is a planned release cadence, not a guarantee that a new version
will be published every month. Publication depends on release readiness and
successful validation. The repository owner communicates any deferral or
change to the schedule.

## Versioning strategy

`llm-graph` uses the three-part version format **`x.y.z`**, commonly written as
**`MAJOR.MINOR.PATCH`**. The third component, sometimes called a sub-minor
version, is conventionally known as the **patch version**.

| Component | Purpose | Backward compatibility |
|---|---|---|
| **Major (`x`)** | Significant changes, including changes to public contracts or architecture. | May or may not remain compatible with earlier major versions. Breaking changes and migration requirements are documented. |
| **Minor (`y`)** | Feature enhancements and additions, such as new supported adapters, views, or output capabilities. | Preserves existing documented public contracts within the same major version, subject to the pre-1.0 policy below. |
| **Patch (`z`)** | Bug fixes and security fixes, without introducing new features. | Preserves documented public contracts within the same minor release line. Corrections to erroneous behavior may change affected results. |

For example:

- `1.2.3` to `2.0.0`: a major release; compatibility must be checked against the
  release notes and migration guidance.
- `1.2.3` to `1.3.0`: a minor release adding backward-compatible functionality.
- `1.2.3` to `1.2.4`: a patch release correcting bugs or security issues.

A major increment resets the minor and patch components to zero. A minor
increment resets the patch component to zero. The highest-impact change in a
release determines its version increment; a minor or major release may also
include bug and security fixes.

### Automated version checks

The publishing workflow requires the GitHub Release tag to match the package
version: version `1.2.3` must use tag `v1.2.3`. The candidate must be strictly
newer than every version with published files in the project's PyPI history,
including yanked releases. Equal or older versions are rejected.

Versions are compared numerically using Python packaging version rules, not
alphabetically. The workflow checks before building and again immediately before
uploading. Network or response errors stop publication; a missing PyPI project
(HTTP 404) permits an initial release. A missing project does not establish
ownership or reserve its name.

Publishing jobs are serialized within this repository. The checks do not
provide an atomic lock against uploads from other repositories or manual
publishers. This policy intentionally prevents older maintenance-line releases
after a newer version has been published.

## Backward compatibility

Compatibility commitments apply to documented public APIs, supported
configuration options, serialized report contracts, and published error codes.
Private helpers and undocumented implementation details are not stable
extension points.

Within a stable major version:

- Minor and patch releases preserve existing public imports, supported call
  patterns, configuration meanings, and documented return contracts.
- New optional functionality does not require existing callers to change their
  code. Additive report metadata is identified in the release notes so consumers
  can accommodate documented schema extensions.
- Error codes retain their published meaning and are not reassigned to unrelated
  failures. Human-readable wording and remediation may be clarified.
- Intentional breaking changes, including removal of documented APIs or
  supported runtimes, require a major release and migration guidance.
- Planned removals are announced through deprecation notices before removal.
  The affected API, recommended replacement, and intended removal version are
  identified where known.

Patch fixes may correct diagram topology, dimensions, metadata, escaping, or
validation that was previously incorrect. Preserving a bug is not a compatibility
guarantee; affected behavior and any necessary user action are explained in the
release notes. If a fix requires breaking a documented public contract, it is
versioned as a breaking change rather than silently shipped as a patch.

Compatibility does not imply byte-for-byte identical diagrams or support for
every version of an upstream model implementation. Each release identifies its
tested Python, PyTorch, Transformers, and model compatibility. New dependency
support is established through validation, not dependency resolution alone.

### Pre-1.0 releases

The current `0.y.z` series is alpha software. Before `1.0.0`, minor releases may
include breaking changes as public contracts mature. Such changes are explicitly
identified in the release notes with migration guidance. Patch releases remain
limited to compatible bug and security fixes within the same `0.y` release line.

The stable-major compatibility commitments take full effect at `1.0.0`.

## What to expect in release notes

The repository owner prepares and approves release notes for each published
version. Readers can expect:

- The version, publication date, and a summary of user-visible changes.
- Feature additions and enhancements, bug fixes, and security fixes, as applicable.
- Changes to model coverage and tested dependency or runtime compatibility.
- Deprecations, breaking API or report-schema changes, and migration guidance.
- Known limitations and a summary of relevant release validation.
- Links to related issues or pull requests where available.

Security-related notes describe impact, affected versions, and upgrade guidance
to the extent appropriate for responsible disclosure. Release notes distinguish
available functionality from planned work.

## Release quality and corrections

Published releases are expected to pass the applicable tests, type checks,
distribution metadata checks, and installation validation. Model-support claims
are limited to validated implementations. Meta-model structural tests do not
establish live tensor values, measured memory usage, or universal model support.

Published artifacts are not replaced in place with different contents.
Corrections are issued under a new version. If a defective release is withdrawn
or yanked, the repository owner communicates the affected version and recommended
upgrade path.

This policy defines upgrade compatibility, not a promise of long-term maintenance
or backported fixes for every earlier major or minor release. Any extended
maintenance commitment is announced separately by the repository owner.
