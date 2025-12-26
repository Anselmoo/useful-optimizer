---
phase: implementation-planning
estimate_points: 20
owner: @you
acceptance_tests:
  - uv run pytest -q -k benchmark_quick
  - python scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json benchmarks/output/*.json
mcp_calls:
  - grep_search: "discovery grep_search '>>>' across repo — ~200 matches; saved discovery/files.json"
  - dependency_audit: "mcp_ai-agent-guid_dependency-auditor — no immediate issues"
  - iterative_coverage: "mcp_ai-agent-guid_iterative-coverage-enhancer — coverage gaps identified"
  - security_review: "mcp_ai-agent-guid_security-hardening-prompt-builder — pre-commit and artifact checks recommended"
  - architecture_design: "mcp_ai-agent-guid_architecture-design-prompt-builder — architecture doc produced"
---

# Plan: Replace trivial doctests with spec-driven benchmarks

## Phases

### Phase 0 — Discovery (complete)
Deliverables: `.github/analysis/discovery/files.json`, `discovery.md`

### Phase 1 — Design & Implementation (in progress)
Deliverables: `architecture.md`, `opt/benchmark/utils.py`, `opt/abstract_optimizer.py::benchmark`, `scripts/replace_trivial_doctests.py`, `tests/` updates, CI workflows.

### Phase 2 — Validation & Rollout
Deliverables: `benchmarks/output/` artifacts, nightly `benchmarks-full` job, `runbook.md`, `rollback-commands.md`.

## Gates & Acceptance
- All `benchmark_quick` tests pass on PRs
- Pre-commit `validate-benchmark-json` hook present and passing on modified artifacts
- `mcp_ai-agent-guid_guidelines-validator` score >= 75/100 (or documented exception)

## Estimation & Sprint Cadence
- **Assumptions**: 1 engineer @ ~80% capacity; story point ≈ 1 day of focused work.
- **Sprint cadence**: 2-week sprints; daily standups; sprint planning and retrospectives at end of each sprint.

| Sprint | Duration | Primary goals | Est. points |
|--------|----------|---------------|-------------|
| Sprint 1 | 2 days | Implement export helpers, unit tests (T001) | 4 |
| Sprint 2 | 3 days | Add `benchmark()` and unit tests (T002) | 6 |
| Sprint 3 | 3 days | Replace trivial doctests (dry-run) and add quick tests (T003,T009) | 6 |
| Sprint 4 | 4 days | CI & pre-commit, nightly job, runbook (T004,T005,T007,T010) | 8 |

**Feedback loops & iteration**
- After each sprint run a short retrospective and update story points and thresholds based on observed velocity and feedback.
- Revalidate plan and tasks with `mcp_ai-agent-guid_guidelines-validator` after sprint 1 and sprint 3 to ensure compliance and reduce risk.

## Risk Register
- Non-determinism in some algorithms — mitigation: use `np.random.default_rng(self.seed)` and small budget quick tests.
- Artifact bloat — mitigation: `track_history` opt-in and compression + size cap enforcement.

## Next actions
- T001..T010 tasks created in `tasks.md` (see tasks.md)
