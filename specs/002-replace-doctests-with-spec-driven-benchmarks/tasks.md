# Tasks: Replace trivial doctests with spec-driven benchmark tests

**Feature Branch**: `feat/benchmarks-specs-ci`
**Spec**: `specs/002-replace-doctests-with-spec-driven-benchmarks/spec.md`
**Plan**: `specs/002-replace-doctests-with-spec-driven-benchmarks/plan.md`

---

## Phase: Setup

- [x] [T001] [ ] [Setup] Implement `opt/benchmark/utils.py` with `export_benchmark_json` and `validate_benchmark_json` — Owner: @you — Est: 2 pts — Files: `opt/benchmark/utils.py`, `tests/test_benchmark_export.py` — Acceptance: `pytest tests/test_benchmark_export.py` passes. (Artifact: `benchmarks/output/` writer + schema validator)

## Phase: Foundational

- [ ] [T002] [ ] [Foundational] Add `benchmark()` to `opt/abstract_optimizer.py` and `opt/multi_objective/abstract_multi_objective.py` — Owner: @you — Est: 4 pts — Files: `opt/abstract_optimizer.py`, `opt/multi_objective/abstract_multi_objective.py` — Dependencies: T001 — Acceptance: `uv run pytest -q -k benchmark_quick` passes for exported artifacts and returns `{'path','metadata'}`.

- [ ] [T009] [P] [Testing] Add `tests/test_benchmarks_quick.py` and parameterize representative optimizers (seeded, shape & reproducibility checks) — Owner: @you — Est: 3 pts — Files: `tests/test_benchmarks_quick.py` — Dependencies: T001, T002 — Acceptance: quick suite completes <1s and passes on CI; reproducibility asserted with `np.allclose(..., atol=1e-12)`.

## Phase: User Stories (P1 first)

- [ ] [T003] [P] [US1] Implement `scripts/replace_trivial_doctests.py` (dry-run & apply) to produce suggested docstring replacements using `optimizer.benchmark(store=True)` examples — Owner: @you — Est: 6 pts — Files: `scripts/replace_trivial_doctests.py` — Dependencies: T001, T002 — Acceptance: `tests/test_doctest_replacements.py` passes; dry-run produces a replacement report. MCPs: run `mcp_ai-agent-guid_gap-frameworks-analyzers` to prioritize replacements and log via `mcp_serena_write_memory`.

- [ ] [T006] [P] [US1] Update docstrings and `docs/` with the in-spec minimal example and 'How to validate locally' snippets — Owner: @docs — Est: 2 pts — Files: `opt/*`, `docs/*` — Dependencies: T003 — Acceptance: `grep -R "benchmark(store=True)" docs/` finds examples; docs build passes.

## Phase: CI & Pre-commit

- [ ] [T007] [ ] [Pre-commit] Add `validate-benchmark-json` hook and `scripts/validate_benchmark_json.py` (used by pre-commit & CI) — Owner: @you — Est: 1 pt — Files: `scripts/validate_benchmark_json.py`, `.pre-commit-config.yaml` — Dependencies: T001 — Acceptance: `pre-commit run validate-benchmark-json --files benchmarks/output/*.json` passes locally.

- [ ] [T008] [ ] [Security] Add `scripts/check_forbidden_mcp_calls.py` and pre-commit hook to block `mcp_` patterns in runtime `opt/` code — Owner: @you — Est: 1 pt — Files: `scripts/check_forbidden_mcp_calls.py`, `.pre-commit-config.yaml` — Acceptance: `python scripts/check_forbidden_mcp_calls.py opt/` exits non-zero if forbidden patterns found.

- [ ] [T004] [ ] [CI] Add `.github/workflows/benchmarks-quick.yml` that runs `uv run pytest -q -k benchmark_quick` on PRs and runs `python scripts/validate_benchmark_json.py` on produced artifacts — Owner: @you — Est: 2 pts — Files: `.github/workflows/benchmarks-quick.yml` — Dependencies: T009, T007 — Acceptance: CI job appears and a test branch shows passing checks.

- [ ] [T005] [ ] [CI] Add nightly `.github/workflows/benchmarks-full.yml` to run `@pytest.mark.benchmark_full`, compress & upload artifacts to `benchmarks/output/`, then validate and archive — Owner: @you — Est: 3 pts — Files: `.github/workflows/benchmarks-full.yml` — Dependencies: T007, T004 — Acceptance: artifacts are uploaded to Actions and validated against schema; archival step recorded.

## Phase: Docs, Polish & Release

- [ ] [T010] [ ] [Release] Add runbook, rollback commands and nightly archiving docs `ci/archiving.md` and `runbook.md` — Owner: @you — Est: 2 pts — Files: `runbook.md`, `rollback-commands.md`, `ci/archiving.md` — Dependencies: T005 — Acceptance: runbook contains smoke test commands and rollback steps.

- [ ] [T011] [ ] [Polish] Add CI smoke job that executes the checklist commands (non-invasive) on PRs — Owner: @you — Est: 1 pt — Files: `.github/workflows/benchmarks-checklist-smoke.yml` — Dependencies: T004 — Acceptance: smoke job runs and reports pass/fail (non-blocking).

## Cross-cutting & Automation

- Each task that executes an MCP MUST record its call with `mcp_serena_write_memory` (for example, T003 should record `mcp_ai-agent-guid_gap-frameworks-analyzers` summary). Include MCP call info (MCP name, args summary, timestamp, memory path) in task descriptions where relevant.

- Export machine-readable `tasks.json` with fields: `id`, `title`, `owner`, `est`, `files`, `acceptance`, `dependencies`, `parallelizable`, `story` and place it at `specs/002-replace-doctests-with-spec-driven-benchmarks/tasks.json`.



---

## Phase 3 — User Stories (priority order) 🧩

### US1 — Replace trivial doctests in optimizer docstrings (P1)

- [ ] T010 [US1] [P] Run the trivial-doctest replacement script and generate a report of candidate files to update (script output: `reports/trivial-doctest-report.md`) — file: `scripts/replace_trivial_doctests.py`
- [ ] T011 [US1] For each candidate file under `opt/` (e.g., `opt/probabilistic/sequential_monte_carlo.py`, `opt/swarm_intelligence/particle_swarm.py`, `opt/evolutionary/cma_es.py`), replace trivial doctest with seeded example and deterministic assertions; update docstring accordingly — example files: `opt/**/*.py`
- [ ] T012 [US1] For each replaced docstring, add or update a spec file fragment in `specs/` pointing to the module and include 'How to validate locally' (e.g., `specs/002-replace-doctests-with-spec-driven-benchmarks/SMC.spec.md`) — folder: `specs/002-replace-doctests-with-spec-driven-benchmarks/`
- [ ] T013 [US1] Manual review: Have a maintainer review every replacement for semantic correctness and doc quality (PR review checklist) — action: request review

### US2 — Quick tests & reproducibility (P1)

- [ ] T014 [US2] Add or extend `benchmark_quick` tests for each replaced optimizer to assert: shape, type, reproducibility (same seed → identical results), and history presence when `track_history=True` — file: `tests/test_benchmarks_quick.py`
- [ ] T015 [US2] Ensure `scripts/check_trivial_doctests.py` passes (no trivial patterns remain) on branch and in pre-commit — file: `scripts/check_trivial_doctests.py`, `.pre-commit-config.yaml`
- [ ] T016 [US2] Run linting/formatting and pydocstyle/unified validator to ensure docstrings and code adhere to templates — commands: `uv run ruff check opt/ && uv run ruff format opt/` and `pre-commit run -a`

### US3 — Nightly full-run benchmarks & CI (P2)

- [ ] T017 [US3] Add GitHub Actions workflow `benchmark.yml` with `quick` job (PR: runs `benchmark_quick`, linter, `specify check`) and `full` scheduled job (nightly: runs `benchmark_full`, uploads `artifacts/*.json.gz`, validates JSON schema) — file: `.github/workflows/benchmark.yml`
- [ ] T018 [US3] Add artifact upload and schema validation step in nightly job (validate against `docs/schemas/benchmark-data-schema.json`) — referenced in `.github/workflows/benchmark.yml`
- [ ] T019 [US3] Configure nightly job to open an issue or alert when regression detected and retain artifacts for 30 days — workflow config and repo settings

---

## Final Phase — Polish & cross-cutting concerns ✨

- [ ] T020 [P] Update VitePress docs: `docs/algorithms/*` with sample run outputs, 'How to validate locally', and a link to benchmark artifacts — files: `docs/algorithms/*.md`
- [ ] T021 [P] Add reviewer checklist to PR template and ensure reviewers run through `specs/.../checklists/reviewers.md` — file: `.github/PULL_REQUEST_TEMPLATE.md` or PR description
- [ ] T022 Prepare & open single PR `feat/benchmarks-specs-ci` including spec files, tests, scripts, CI changes, and docs; fill PR body with checklist and validation steps — action: open PR on GitHub
- [ ] T023 Post-merge: Monitor first 3 nightly runs, triage regressions (issues opened, assign maintainers), and finalize thresholds after data collected — action: maintainers + CI alerts

---

## Cross-cutting tasks added

- [ ] T024 [ ] [Process] Add PR verification job that enforces: changes to `opt/` must include a spec fragment under `specs/` and at least one `benchmark_quick` test; PR body must include an `MCP calls:` section or structured JSON block. — Owner: @you — Est: 2 pts — Files: `.github/workflows/pr-verify.yml`, scripts: `scripts/check_pr_mcp_and_spec.py` — Acceptance: test branch PR without spec update fails the verification job.

- [ ] T025 [ ] [Tests] Add tests for tasks parity and duplication detection (`tests/test_tasks_consistency.py`) and MCP logging expectations (`tests/test_mcp_logging.py`) — Owner: @you — Est: 1 pt — Files: `tests/test_tasks_consistency.py`, `tests/test_mcp_logging.py` — Acceptance: `pytest tests/test_tasks_consistency.py tests/test_mcp_logging.py` pass locally.

- [ ] T026 [ ] [Refactor] Remove or flag dead/unused code found during coverage analysis — Owner: @you — Est: 1 pt — Files: `opt/*` — Acceptance: Deprecated code removed or flagged, tests pass, and PR includes justification for removals.

- [ ] T027 [P] [Testing] Add targeted quick tests for top discovered modules (MantaRay, `penalty_method`, SineCosine, ParticleFilter) — Owner: @you — Est: 3 pts — Files: `tests/benchmark_quick/test_manta_ray.py`, `tests/benchmark_quick/test_penalty_method.py` — Dependencies: T001, T002 — Acceptance: tests complete <1s and validate artifacts.

- [ ] T028 [ ] [CI] Add CI coverage-analysis job and adaptive thresholds — Owner: @you — Est: 2 pts — Files: `.github/workflows/coverage-analysis.yml` — Acceptance: coverage job runs weekly and suggests threshold updates.

- [ ] T029 [ ] [Release] Tune archival retention policy and update runbook (`ci/archiving.md`, `runbook.md`) — Owner: @you — Est: 1 pt — Files: `ci/archiving.md`, `runbook.md` — Acceptance: runbook contains archival verification steps and retention is enforced in nightly job.


---

## Dependencies & Story Completion Order

1. Phase 1 setup tasks (T001-T004) must be completed before Phase 2.
2. Phase 2 foundational tasks (T005-T009) must be completed before User Stories tasks begin.
3. US1 & US2 tasks should run in parallel where safe (T011 and T014 are related and may be executed together for the same module).
4. CI workflow (T017-T019) should be completed before opening the single PR (T022) to ensure CI checks are available.

## Parallel execution examples [P]

- Developers can implement `benchmark_quick` tests for different algorithms in parallel (e.g., one dev for SMC, another for PSO) — mark tasks as `[P]` where independent
- Docs updates, spec fragments, and test additions for individual algorithms are parallelizable after foundational tasks are in place

## Implementation strategy (MVP first)

- MVP: replace trivial doctests for 3 representative algorithms (SMC, PSO, CMA-ES), add `benchmark_quick` tests for them, add trivial-doctest checker, and add `quick` CI job.
- Iterate: expand replacements + full-run tests, add artifact validation and nightly scheduling, tune thresholds based on data.

---

If you'd like, I can now: 1) create `scripts/replace_trivial_doctests.py` and `scripts/run_benchmark.py` scaffolds, 2) draft `.github/workflows/benchmark.yml`, and 3) generate PR body template for `feat/benchmarks-specs-ci`. Which should I start with?
