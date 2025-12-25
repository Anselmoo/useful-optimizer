<!--
Sync Impact Report
- Version change: none 100 1 --> 0.1.0
- Modified principles: added/BBOB compliance, Test-First, Documentation & Schema Integrity, Observability/Versioning
- Added section: Additional Constraints (tooling & dependency rules)
- Updated templates:
  - `.github/prompts/optimizer-docs-template.prompt.md` ✅ updated (COCO example now records history and saves run)
  - `.github/copilot-instructions.md` ✅ updated (added `uv sync --all-groups --all-extras` guidance)
- Templates requiring review: `.specify/templates/plan-template.md` ⚠ pending
- Follow-up TODOs: RATIFICATION_DATE needs to be set; confirm version bump policy with maintainers
-->

# Useful Optimizer Constitution

## Core Principles

### I. Library-First
Every new feature or algorithm must be developed as a well-scoped, self-contained library component. Libraries MUST be independently testable, documented, and importable via their full path (e.g., `from opt.category.module import Class`). Purpose and scope MUST be explicit in the README and docs; organizational-only libraries (no public API) are discouraged and require explicit justification.

### II. COCO/BBOB Compliance (NON-NEGOTIABLE)
All optimizers that advertise COCO/BBOB compatibility MUST implement: seeded randomization (`seed` parameter), reproducible execution (use `np.random.default_rng(self.seed)`), history tracking (`track_history` flag), and IOHprofiler/COCO-compatible exports (via `benchmarks.save_run_history` or equivalent). Doctest examples intended for benchmarking MUST record history and be usable to generate benchmark artifacts.

### III. Test-First (NON-NEGOTIABLE)
Testing is mandatory before implementation. Unit tests, doctests, and COCO/benchmark examples MUST use fixed seeds where non-determinism exists. Any change that affects optimizer behavior MUST include tests demonstrating the intended behavior and reproducibility. Pre-commit hooks and `uv run ruff check` MUST pass before a PR is ready for review.

### IV. Documentation & Schema Integrity
All optimizer docstrings MUST follow the COCO/BBOB docstring template (`.github/prompts/optimizer-docs-template.prompt.md`) and the schema in `docs/schemas/docstring-schema.json`. Docstrings MUST include the 11 required sections and use LaTeX for mathematical notation. Schema violations are not allowed on merge; automated validation (scripts/unified_validator) MUST pass in CI.

### V. Observability, Versioning & Simplicity
Optimizers MUST provide sensible defaults for observability (e.g., `track_history=False` by default, but implemented), memory-efficient history options (configurable via a `HistoryConfig` dataclass), and structured metadata in benchmark exports. Versioning follows semantic versioning for the project artifacts and documentation: MAJOR for breaking governance changes, MINOR for new principles or policy additions, PATCH for wording/typo clarifications.

## Additional Constraints
- Tooling: Project supports Python 3.10–3.12. Use `uv` to manage environments and dependencies.
- Dependency sync rule: when adding new development packages or extras, run `uv sync --all-groups --all-extras` to install all group/extras and update the lockfile.
- Linting & formatting: `uv run ruff check opt/` and `uv run ruff format opt/` are required before committing. Run `pre-commit run -a` during final checks.
- Performance: Optimizers and benchmark runs MUST be written to complete small validation scenarios quickly (< 1s where appropriate) to keep CI practical.

## Development Workflow
- PRs documenting constitution compliance: Any change that affects principles MUST reference this constitution and include a migration plan and tests.
- Review requirements: PRs MUST include passing CI, a summary of schema / docstring changes, and a short note indicating whether the change requires a constitution version bump.
- Doctests and examples: All doctests used in documentation that are intended to generate benchmark data MUST persist or export data (e.g., using `save_run_history`) so documentation builds and visualizations have concrete inputs.

## Governance
Amendments to this constitution MUST be proposed via a PR targeting `.specify/memory/constitution.md` and include:
- A clear rationale and impact analysis
- A suggested version bump (MAJOR/MINOR/PATCH) with justification
- A migration plan for any affected templates or tooling

Approval requires at least one maintainer + one reviewer approval OR merger by a maintainer after a 72-hour review period. Emergency patches for wording/typos (PATCH level) may be merged with a single maintainer approval but MUST be announced in the next weekly sync.

**Version**: 0.1.0 | **Ratified**: TODO(RATIFICATION_DATE): Please set adoption date | **Last Amended**: 2025-12-25
