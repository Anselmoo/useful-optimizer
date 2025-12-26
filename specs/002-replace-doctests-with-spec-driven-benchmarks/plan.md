# Implementation Plan: Replace trivial doctests with spec-driven benchmark tests

**Branch**: `feat/benchmarks-specs-ci`  | **Date**: 2025-12-25  | **Spec**: `specs/002-replace-doctests-with-spec-driven-benchmarks/spec.md`

## Summary
Replace trivial doctests across `opt/` with seeded, reproducible examples and add a test harness (fast `benchmark_quick` tests for PRs and `@pytest.mark.benchmark_full` tests for nightly validation), memory-efficient history storage (validate PR #119), CI workflows (quick PR checks + scheduled full runs), Spec Kit enforcement (`specify check`), and docs generation. Deliverable: one comprehensive PR containing specs, tests, CI, and docs.

## Phase 0: Design (2–3 days)
Goals
- Produce detailed component designs, interfaces, and data schemas
- Decide quick/full test parameters and thresholds
- Update Spec Kit plans & constitution checks

Tasks
1. Research & decisions (owner: author)
   - Confirm `benchmark_quick` parameters: `max_iter=50`, `dim` default 10 for examples, seed=42, target runtime <1s
   - Confirm `benchmark_full` parameters: `max_iter=2000` (adjust per algorithm), seeds range 0-14 for statistical runs
   - Confirm history artifact format: compressed JSON following `docs/schemas/benchmark-data-schema.json`
2. Component design
   - Benchmark runner: CLI + Python API to execute a single run and save history
   - Memory-efficient history store: ensure PR #119 approach matches schema and bounded memory usage
   - Test harness: pytest markers `benchmark_quick`, `benchmark_full`
   - CI jobs: quick (PR), full (nightly scheduled), artifact upload and validation
3. Acceptance criteria (design)
   - Quick tests must execute in <1s on dev VMs
   - History artifact ≤ 200MB compressed in normal full runs
   - All specs follow Spec Kit templates and pass `specify check`

Artifacts
- `specs/.../plan.md` (this file)
- Mermaid diagram showing data flow and CI pipelines

## Phase 1: Implement (3–7 days)
Goals
- Implement tests, scripts, pre-commit hooks, and CI workflows; replace docstring examples in a programmatic but reviewed way

Tasks (ordered)
1. Add quick tests (done, but expand): add parameterized entries covering representative optimizers
2. Add full-run tests (scaffold present; add more per-algorithm where maintainers pick)
3. Implement `scripts/check_trivial_doctests.py` (done) and include as pre-commit hook (done)
4. Add tests to validate history artifact format and compressed size limit (new tests)
5. Implement benchmark runner CLI (e.g., `scripts/run_benchmark.py`) to produce history artifacts, allow small/full modes, and accept seeds
6. Replace trivial doctest examples across `opt/` with robust seeded examples (automation script + manual review)
7. Add and tune `specify` integration: add spec files to `specs/` and ensure `specify check` runs locally and in CI
8. Update doc pages in `docs/algorithms/*` with sample outputs and benchmarking notes

Acceptance criteria (implement)
- `benchmark_quick` tests pass in <1s on CI PR runners
- `check_trivial_doctests.py` passes (no matches left) before merge
- `specify check` passes in CI
- History artifact tests validate against `docs/schemas/benchmark-data-schema.json`

Artifacts
- `tests/test_benchmarks_quick.py` (complete)
- `tests/test_benchmarks_full.py` (scaffold + selected algorithms)
- `scripts/run_benchmark.py`, `scripts/check_trivial_doctests.py`
- Doc updates and spec files under `specs/`

## Phase 2: Validate (2–4 days, ongoing)
Goals
- Ensure nightly full-run runs, artifacts are uploaded, validated, and regressions are monitored

Tasks
1. CI: Create GitHub Actions workflow `benchmark.yml` with two workflows
   - `quick` (PR): runs `benchmark_quick`, `specify check`, linter; artifacts: small logs
   - `full` (nightly): runs `benchmark_full` tests, compresses and uploads history JSON (`.json.gz`) and checks schema
2. Monitoring & alerts
   - If nightly run fails acceptance thresholds, fail workflow and open an issue with context and links to artifacts
3. Run first 3 nightly runs and tune thresholds (reduce flakiness)
4. Add docs on how to add new optimizer tests and how to run both modes locally

Acceptance criteria (validate)
- Nightly runs complete and artifacts validate against schema within a week of rollout
- Any regressions result in issues and are triaged within 48 hours as per Constitution
- CI enforces `specify check` and `check_trivial_doctests.py` on PRs

## One-Big-PR workflow
- Branch: `feat/benchmarks-specs-ci`
- Work locally in branch; use `uv sync` and `uv run pytest -q -k benchmark_quick` to test
- Commit message guidance: `feat: replace trivial doctests with spec-driven tests + CI` (use conventional commit style)
- Draft PR description: include summary, list of changes (specs, tests, scripts, CI), validation steps, and checklist (from spec list)
- Request at least 1 maintainer review, ensure CI is green before merge

## Rollback plan
- If post-merge regressions occur: revert PR branch (create `revert/<pr-number>`), open issue with artifacts and revert notes, reintroduce changes on a follow-up branch with fixes
- Keep PR atomized via feature toggles (e.g., disable full nightly temporarily via workflow inputs) if immediate rollback is needed

## Effort Estimates & Risks
- Phase 0 (Design): 2–3 days (one engineer)
- Phase 1 (Implement): 3–7 days (one engineer) — more if replacing many doctests and implementing many full-run tests
- Phase 2 (Validate): 2–4 days (monitoring + tuning)

Risks & Mitigations
- Flaky tests: mitigate by conservative thresholds, reproducible seeds, repeat runs to stabilize thresholds
- Large artifact sizes: mitigate by memory-efficient history structure (PR #119) and sampling/compression strategies
- CI time budget: ensure quick tests are small; schedule full-run at night and allow longer runtime

## Next steps
1. Approve plan and add `plan.md` to `specs/002-replace-doctests-with-spec-driven-benchmarks/` (done)
2. Implement remaining test coverage and history validation tests (Phase 1)
3. Create GitHub Actions workflow for quick + nightly full runs (Phase 2)
4. Open `feat/benchmarks-specs-ci` as one big PR

---

**Owner**: @maintainers
**Status**: Ready for `/speckit.tasks` to expand into task-level details and assignment
