<!-- Sync Impact Report
Version change: none → 1.0.0
Modified principles: Added Test-First (TDD); Spec-Driven Development; Reproducible Benchmarks; Minimal Doctests; Memory-Efficient History Tracking; CI-enforced spec checks; Continuous Improvement & Iteration; Performance Best Practices (early_stop & max_iter guidance)
Added sections: MCP Usage, Artifacts & Versioning, Review Criteria, Project Planning & Sprints, Performance Best Practices
Templates requiring updates: .specify/templates/plan.md ⚠ pending; .specify/memory/constitution.md ✅ updated; .github/prompts/spec-kit.prompt.md ✅ updated
Follow-ups: Ensure .github/checklists/benchmarks-spec.md exists; add CI job to archive benchmarks nightly; add pre-commit script to flag unexplained `max_iter>=5000`
-->

# Useful Optimizer Constitution

**Governance:** Version: 1.0.0 | Ratified: 2025-12-26 | Last Amended: 2025-12-26

## Purpose
Provide compact, enforceable rules to ensure high-quality, reproducible benchmarks and spec-driven development for the useful-optimizer project.

## Principles
- **Test-First (TDD):** All new features and bug fixes MUST be accompanied by tests that cover the intended behavior before implementation.
- **Spec-Driven Development:** Specs (spec/*.md or .specify artifacts) MUST be the primary source of truth for design and acceptance criteria.
- **Reproducible Benchmarks (seeded, archived):** Benchmarks MUST be deterministic by seeding RNGs; full runs MUST include seeds 0-14 for coverage. Example line: `Reproducible Benchmarks: seeds 0-14 required for full runs`.
- **Minimal Doctests (replace with mini-benchmarks):** Replace long or brittle doctests with focused mini-benchmarks in `benchmarks/quick` or `benchmarks/mini`.
- **Memory-Efficient History Tracking (track_history opt-in):** Opt-in only: optimizers MUST write history only when `track_history=True`; large histories MUST be avoided unless justified.
- **CI-enforced spec checks:** CI MUST run spec validation and `validate-benchmark-json` checks on all PRs touching optimizers or benchmark artifacts.
- **Performance Best Practices:** Benchmarks and optimizer implementations MUST support early stopping (`early_stop` or `stop_threshold`) and include `metadata.iterations`, `metadata.stopped_early` (bool), and `metadata.stopping_reason` (str). Quick benchmarking defaults MUST be conservative (e.g., `max_iter <= 500`); **do not use** `max_iter=10000` as a default in benchmark comparisons without documented justification. Add audit scripts and pre-commit checks as required.
- **Continuous Improvement & Iteration:** Specs and implementations MUST include short feedback cycles (reviews/retrospectives) and evidence of iteration when required.

## Enforcement
- **All PRs modifying optimizers MUST** include:
  - a corresponding spec update (or a justification why none is needed),
  - at least one `benchmark_quick` test demonstrating the change, and
  - a passing `validate-benchmark-json` pre-commit check or CI job.
- **Pre-commit & Audit**: The repository MUST include a pre-commit script that flags unexplained `max_iter>=5000` literals in code, tests, and docs; PRs that intentionally increase default `max_iter` must include a performance rationale and comparative evidence.

## MCP Usage
All spec, plan, and implementation steps MUST call (and log in the PR description):
- `mcp_context7_get-library-docs` (use the Spec Kit template),
- `mcp_ai-agent-guid_prompt-chaining-builder`,
- `mcp_ai-agent-guid_guidelines-validator`,
- `mcp_serena_*` (activate/find/insert/replace where needed), and
- GitHub MCPs used to create branches and PRs.

Document MCP calls in PR descriptions with a short 'MCP calls:' list for traceability.

## Artifacts & Versioning
- Benchmark artifacts MUST be stored as `benchmarks/output/<algorithm>-<timestamp>-s<seed>.json`.
- Artifact JSON MUST include `schema_version` in its top-level metadata.
- Nightly CI MUST archive benchmarks into an artifacts bucket and rotate at retention policy defined in `ci/archiving.md`.

## Review Criteria
PRs touching optimizers or benchmark artifacts MUST include:
- A short gap-analysis generated via `mcp_ai-agent-guid_gap-frameworks-analyzers` (1-3 bullets),
- A checklist referencing `.github/checklists/benchmarks-spec.md`, and
- Evidence of MCP validation steps and green CI jobs for `benchmark_quick` and `validate-benchmark-json`.

## Project Planning & Sprints
- Plans and specs MUST include a high-level timeline (milestones and ETA), a stated review cadence, and evidence of capacity planning.
- Use data-driven estimates where possible and add a short 'Constitution Check' to each plan using Spec Kit gates.

## Generator Prompt (example - use verbatim)
"Using the Spec Kit constitution template (library `/github/spec-kit`), draft a short Constitution file for the 'useful-optimizer' project: include Principles, Enforcement rules, MCP usage requirements, Artifact naming/versioning rules, and Review Criteria (3-5 bullets each). Validate the text using `mcp_ai-agent-guid_guidelines-validator` to ensure it follows repository doc standards and output a markdown file content ready to save at `.github/CONSTITUTION.md`."

## Compliance & Amendments
- Amendments MUST be applied by PR that updates this file and must include a semantic version bump. Major/minor/patch decisions MUST follow the standard described in `.specify/memory/constitution.md`.

<!-- End of Constitution -->
