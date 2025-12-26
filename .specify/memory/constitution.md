<!--
Sync Impact Report:
- Version change: TEMPLATE → 0.1.0
- Modified/Added principles: Added I. Test-First (TDD); II. Spec-Driven Development; III. Reproducible Benchmarks; IV. CI-Enforced Spec Checks; V. Observability & Memory-Efficient History Tracking
- Added sections: Additional Constraints (Performance & Storage); Development Workflow
- Templates requiring updates: .specify/templates/spec-template.md (✅ updated), .specify/templates/plan-template.md (⚠ pending), .specify/templates/tasks-template.md (⚠ pending), .github/prompts/spec-kit.prompt.md (✅ updated)
- Follow-up TODOs: Run `mcp_ai-agent-guid_guidelines-validator` (performed); run `specify check` in CI; add checklist entries for benchmark tests; confirm RATIFICATION_DATE and gather maintainer approval.
-->

# Useful Optimizer Constitution

## Core Principles

### I. Test-First (TDD) (NON-NEGOTIABLE)
All new behavior MUST be specified by tests before code is added. Tests are the primary artifact for correctness and must be present in the same change that implements the behavior. For algorithmic or benchmark changes, tests MUST include:
- Fast sanity tests (`benchmark_quick`) suitable for PR feedback (target: <1s runtime)
- Integration/benchmark tests (`@pytest.mark.benchmark_full`) for nightly or scheduled validation
- Reproducibility checks (fixed seeds, deterministic assertions where possible)
Rationale: Prevent regressions and ensure reliability of algorithm implementations and benchmarks.

### II. Spec-Driven Development (MANDATORY)
Every feature, bug-fix, or behavioral change MUST have a Spec Kit feature specification (use `/specify` templates). Specs MUST include clear Given/When/Then scenarios, edge cases, measurable acceptance criteria, and 'How to validate locally' steps. The spec is the ground-truth for tests and PR descriptions.
Rationale: Aligns developers and reviewers on expected behavior and reduces ambiguous tests (e.g., trivial doctests with no acceptance criteria).

### III. Reproducible Benchmarks (REQUIRED)
Benchmarks MUST be reproducible and seedable. Records for each benchmark run MUST include: seed, environment (Python & dependency versions), runtime parameters, and a memory-efficient history artifact (see History policy). Benchmarks MUST provide both quick validation modes and full-run modes for robust evaluation.
Rationale: Ensure experiments are comparable and debugging is practical across contributors and CI.

### IV. CI-Enforced Spec Checks & Testing Gates
All PRs MUST run automated checks that include: fast sanity tests, linting, and `specify check` for spec compliance. The repository MUST run scheduled full benchmark jobs (nightly or weekly) that execute `@pytest.mark.benchmark_full` tests, store artifacts, and fail if regressions in acceptance metrics are detected.
Rationale: Automate enforcement of the constitution and catch regressions early.

### V. Observability & Memory-Efficient History Tracking
All benchmark runs MUST produce structured, schema-driven history artifacts stored in a memory-efficient format (see `benchmarks/schemas/benchmark-history.json` or similar). History artifacts MUST be versioned, compressed where appropriate, and retained according to storage policy. PR #119's memory-efficient history implementation is adopted as the recommended approach and MUST be validated by tests that assert correctness and bounded memory usage.
Rationale: Observability and compact histories make it possible to analyze trends and validate algorithm behavior without excessive storage costs.

## Additional Constraints (Performance, Storage, and Reproducibility)
- **Quick Tests**: Target <1 second execution per quick benchmark test when run on typical development machines.
- **Full Runs**: Nightly full-runs may be lengthy; CI artifacts MUST be uploaded and retained for at least 30 days.
- **Memory**: A typical full-run history artifact SHOULD not exceed 200MB compressed; any exception MUST be documented in the plan and approved.
- **Schemas**: Use explicit JSON schemas for all artifacts; link to schema files in the plan and the docs.

## Development Workflow, Review Process & Quality Gates
- **Branching**: Feature branches MUST follow `feat/<short-description>` or `fix/<short-description>`.
- **One-Big-PR Guidance**: For this initiative (replacing trivial doctests and finishing PR #119), a single branch and PR is preferred. The PR MUST include:
  - Spec Kit feature spec(s) attached or linked
  - `benchmark_quick` tests passing locally
  - `@pytest.mark.benchmark_full` tests marked and documented (run in CI nightly)
  - CI changes for quick & full benchmark jobs
  - Documentation updates and 'How to validate locally' instructions
- **Iterative delivery & sprint planning**: Work should be organized into short, timeboxed iterations (e.g., 1–2 week sprints) with clear milestones, acceptance criteria, and planned reviews. Use `/speckit.plan` and `/speckit.tasks` to create sprint-scoped plans and the `mcp_ai-agent-guid_sprint-timeline-calculator` to help sequence tasks and estimate timelines. Each iteration MUST include a review and a short retrospective to incorporate feedback quickly. Each sprint MUST include a mid-sprint demo and an end-of-sprint demo plus a retrospective; feedback collected during demos and retrospectives MUST be prioritized and scheduled into subsequent sprints. Feedback from demos, users, or CI (including benchmark regressions) MUST be filed as actionable issues and triaged within 48 hours. Plans MUST include time estimates, a risk buffer (suggested 15-25%), and a proposed completion date; use `mcp_ai-agent-guid_sprint-timeline-calculator` and data-driven estimation when possible. Teams MUST track basic velocity metrics and adjust scope accordingly.
- **Reviewers**: At least one maintainer review is REQUIRED for code and spec changes; CI must be green before merging.

## Governance
- Amendments to this Constitution MUST be proposed via a PR that includes: a migration plan, rationale, and any test/template changes.
- **Approval**: A non-authoring maintainer approval (1 approval) is required; for major governance changes (MAJOR version bump) a two-thirds consensus of active maintainers is recommended.
- **Versioning**: Follow semantic versioning for the constitution document:
  - MAJOR: Backward-incompatible governance/principle removals or redefinitions
  - MINOR: New principle or material expansion of guidance
  - PATCH: Clarifications, wording fixes, templates-only changes

**Version**: 0.1.0 | **Ratified**: 2025-12-25 | **Last Amended**: 2025-12-25

```
