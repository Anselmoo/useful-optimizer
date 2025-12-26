# Benchmarks Gap Report

Date: 2025-12-26

Summary of gaps discovered for replace-doctests-with-spec-driven-benchmarks feature:

- Trivial doctests exist in ~200 places (shape checks, simple isinstance checks).
- Few `benchmark_quick` tests present; majority of optimizers do not have quick sanity checks that export artifacts.
- No pre-commit hook to validate `benchmarks/output/*.json` artifacts on commit.
- CI lacks nightly `benchmark_full` archival job and artifact validation.

Top prioritized actionable tasks:
- T001: Add `export_benchmark_json` & `validate_benchmark_json` helpers (high)
- T002: Add `benchmark()` to `AbstractOptimizer` and mirror in multi-objective (high)
- T003: Implement `scripts/replace_trivial_doctests.py` to produce replacement suggestions (medium)
- T004: Add `benchmark_quick` tests for representative optimizers and parametrize (high)
- T005: Add `validate-benchmark-json` pre-commit hook (high)
- T006: Add GitHub Actions quick & full workflows (high)

Full report and action items saved under `.github/analysis/` and will be referenced in `plan.md` and `tasks.md`.

## Machine-readable report
- File: `.github/analysis/benchmarks-gap-report.json` (contains prioritized tasks and categorizations)

## New/Updated Action Items (added to spec tasks list)
- T027: Add targeted quick tests for top discovered modules (MantaRay, penalty_method, SineCosine, ParticleFilter) — **High**
- T026: Remove or flag dead/unused code found during coverage analysis — **Medium**
- T028: Add CI coverage-analysis job and adaptive thresholds — **Medium**
- T029: Tune archival retention and update runbook (`ci/archiving.md`, `runbook.md`) — **Medium**

Please see `.github/analysis/benchmarks-gap-report.json` for structured details and priorities.
