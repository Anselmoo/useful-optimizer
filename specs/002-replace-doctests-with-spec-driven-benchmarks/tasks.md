# Tasks: Replace trivial doctests with spec-driven benchmark tests

**Feature Branch**: `feat/benchmarks-specs-ci`
**Spec**: `specs/002-replace-doctests-with-spec-driven-benchmarks/spec.md`
**Plan**: `specs/002-replace-doctests-with-spec-driven-benchmarks/plan.md`

---

## Phase 1 — Setup ✅

- [ ] T001 [P] Create feature branch `feat/benchmarks-specs-ci` and prepare environment (repo root): `git checkout -b feat/benchmarks-specs-ci` && `uv sync`
- [ ] T002 [P] Ensure Spec Kit is initialized and global spec is present (`specs/002-replace-doctests-with-spec-driven-benchmarks/spec.md`) — verify `specify init . --force` and `specify check` pass
- [ ] T003 [P] Add trivial doctest checker script and pre-commit entry (`scripts/check_trivial_doctests.py`, `.pre-commit-config.yaml`)
- [ ] T004 [P] Add benchmark runner CLI scaffolding (`scripts/run_benchmark.py`) to execute small/full runs and emit compressed history artifacts (`artifacts/*.json.gz`)

---

## Phase 2 — Foundational tasks (blocking for stories) 🔧

- [ ] T005 [P] Implement `benchmark_quick` tests (`tests/test_benchmarks_quick.py`) with parameterized seedable runs for representative optimizers (SMC, PSO, CMA-ES) — file: `tests/test_benchmarks_quick.py`
- [ ] T006 [P] Implement `@pytest.mark.benchmark_full` scaffolds for nightly runs (`tests/test_benchmarks_full.py`) — file: `tests/test_benchmarks_full.py`
- [ ] T007 [P] Add history artifact validation tests (`tests/test_history_artifacts.py`) that validate schema and compressed size (<= 200MB guideline) — file: `tests/test_history_artifacts.py`
- [ ] T008 [P] Add/verify unified docstring validator & dependencies (pre-commit: `unified-docstring-validator`, `pyproject.toml` entries) — files: `.pre-commit-config.yaml`, `pyproject.toml`
- [ ] T009 Add `scripts/replace_trivial_doctests.py` to locate trivial patterns and produce replacement report (manual review required) — file: `scripts/replace_trivial_doctests.py`

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
