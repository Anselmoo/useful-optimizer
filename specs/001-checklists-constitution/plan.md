# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This feature replaces fragile inline doctests that only assert trivial output with **spec-driven benchmark tests** that (a) run quick sanity checks in PRs and (b) can be executed (via CLI or doctest-enabled examples) to produce reproducible benchmark artifacts used by the docs. The goal is a 3-phase delivery (Design → Implement → Validate) producing the following modules: benchmark runner, memory-efficient history store, test harness (quick + full), CI workflows (quick PR checks + scheduled full-run), Spec Kit checks, docs generation from benchmark artifacts, and observability/performance goals. The final delivery will be validated on GitHub with end-to-end tests and an explicit rollback plan for the single-PR workflow.

## Technical Context

**Language/Version**: Python 3.10–3.12 (target 3.12)
**Primary Dependencies**: numpy, pytest, click (for CLI), jsonschema, pydantic (for schema validation), gzip/zstd (for compressed artifacts), COCO/BBOB tooling (optional for full runs)
**Storage**: Compressed JSON (gz or zst) artifact files stored under `benchmarks/results/<run-id>/` and uploaded as CI artifacts; minimal local store `opt/history` for in-memory/memory-mapped history during runs.
**Testing**: pytest with custom markers `benchmark_quick` and `benchmark_full`; use `@pytest.mark` to separate quick vs full runs; test harness will include deterministic seeds and mutable-run toggles.
**Benchmarking**: Quick sanity tests (<1s) must run on PRs (single-run with small iter/constraints); nightly scheduled full COCO/BBOB runs run in GitHub Actions or a scheduled machine with full evaluation budget (dim × 10000 evals, 15 seeds). per-run history artifacts MUST be stored as compressed JSON and validated against `schemas/run-artifact-schema.json`, and aggregate/nightly results must conform to the docs schema at `docs/schemas/benchmark-data-schema.json`. Artifacts must include metadata (commit sha, run-id, seed, function, dim, bounds, runtime, memory, best_fitness history).
**Target Platform**: Linux (CI) and macOS (local dev)
**Project Type**: Library (single codebase)
**Performance Goals**: Quick tests complete in <1s on GitHub-hosted runners; history store uses <100MB memory for quick tests; full-run artifacts compressed to <50MB per function/run where feasible; observability: track runtime and memory per run (p95 < threshold TBD)
**Constraints**: Deterministic seeds required for reproducible runs; avoid committing large binary blobs to repo—use CI artifact storage or an external bucket if necessary.
**Scale/Scope**: Affects doctests and small benchmark harness in the repo; a limited set of representative optimizers (e.g., `MantaRayForagingOptimization`) will be converted first; later expand to other algorithms if successful.

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

- Must preserve single-line docstring doctest style where possible and provide clear migration instructions.
- Must include Spec Kit checks in CI that enforce: spec exists for every replaced doctest, `benchmark_quick` tests present and passing, reproducibility documented.
- Storage and artifact retention policy must be documented and approved.

## Project Structure (proposed)

Repository changes and new modules:

``text
opt/benchmark/runner.py # CLI + programmatic runner for quick/full benchmarks
opt/benchmark/**init**.py
opt/history/store.py # Memory-efficient history store (gzip/zstd + optional mmap)
schemas/benchmark-data-schema.json# JSON schema for compressed history artifacts
tests/benchmark_quick/ # Quick PR-focused benchmark tests (fast, deterministic)
tests/benchmark_full/ # Full BBOB/COCO integration tests (scheduled)
docs/benchmarks/ # Docs pages generated from benchmark artifacts
.github/workflows/benchmark-quick.yml # Quick PR checks
.github/workflows/benchmark-nightly.yml# Scheduled full runs, artifacts upload
.specs/001-benchmarks-specs/ # Spec & plan artifacts for this feature (spec.md/research.md/plan.md)

```

**Structure Decision**: Use the existing `opt/` package as the canonical place for the runner and the history store to keep imports simple (`from opt.benchmark import run`). Tests live under `tests/benchmark_*` and use pytest markers.

## Phased Implementation Plan

### Phase 0 — Design & Research (deliverables: `research.md`)
- Resolve open questions (see Research Tasks below): artifacts schema fields, compression format (gz vs zstd), quick-test paramization (max_iter small values), CI quota & cost for nightly runs, artifact retention policy.
- Produce `schemas/benchmark-data-schema.json` and a minimal `data-model.md` describing artifact contents.
- Produce acceptance criteria and test definitions for `benchmark_quick` and `benchmark_full` (time, artifact size, reproducibility checks).
- Produce the Spec Kit check definitions (what `specify check` looks for).

### Phase 1 — Implement (deliverables: code + docs + tests)
- Implement `opt.benchmark.runner` with CLI (`python -m opt.benchmark.runner --function sphere --dim 10 --seed 42 --out=...`) and programmatic API.
- Add memory-efficient history store in `opt.history` with optional compressed streaming write and schema validation using `pydantic` + `jsonschema`.
- Replace selected doctests (starting with `MantaRayForagingOptimization`) with new spec-driven quick tests under `tests/benchmark_quick/` that assert: runs complete, outputs artifact file, artifact validates against schema, and `len(solution) == dim` (as before) and `fitness` numeric.
- Add test harness hooks to generate docs snippets from the produced artifacts (e.g., `docs/benchmarks/generated/...`), with a small script to generate markdown pages from artifacts.
- Add `specs/001-benchmarks-specs/*` artifacts: `spec.md`, `research.md`, `data-model.md`, `contracts/` (if any), and `quickstart.md` describing how to run locally and generate docs.
- Add ruff/formatting and update docstrings to reference new workflows.

### Phase 2 — Validate & CI integration (deliverables: CI workflows, docs, and GitHub verification)
- Add `.github/workflows/benchmark-quick.yml` to run `pytest -m benchmark_quick -q` and `ruff` on PRs (fast checks). Mark these as required in branch protection for the branch used for the final PR.
- Add `.github/workflows/benchmark-nightly.yml` scheduled job to run `pytest -m benchmark_full` (or dedicated runner), upload compressed artifacts to Actions artifacts storage, and run a post-check job that validates artifact schema and basic metrics. Notify maintainers on failures.
- Add a docs-generation job that consumes benchmark artifacts and updates `docs/benchmarks/generated/` pages (as a draft artifact to review in PR or a docs branch for publication).
- Test the entire workflow end-to-end on GitHub: open a single 'one-big-PR' (see below), ensure quick checks run, then merge and confirm scheduled job can run and artifacts are produced.

## One-Big-PR Workflow & Rollback Plan

**One-Big-PR workflow** (intended for coordinated migration of doctests to spec-driven benchmarks):
1. Create branch `feat/benchmarks-specs` or `001-benchmarks-specs` (we used `001-benchmarks-specs-ci` for plan creation).
2. Incrementally add commits (small, logical changes) covering: spec files, schema, runner, history store, small quick tests, docs-generation script, and CI workflow files.
3. Run `uv sync`, `uv run ruff check`, and `uv run pytest -m benchmark_quick -q` locally; include quick benchmark artifacts for review.
4. Open a single comprehensive PR with the checklist (Spec exists for each doctest replaced, quick tests pass, schema validated, CI added).

**Rollback plan**:
- If quick checks fail on PR, fix in PR before merging.
- If regressions are discovered post-merge (e.g., docs generation or nightly runs fail), revert the merge commit immediately (use the GitHub revert interface) or open a follow-up fix PR targeted at the failing area.
- To minimize blast radius, gating rules in CI: docs generation step is opt-in or gated behind a flag (e.g., `GENERATE_BENCH_DOCS=true`) until we confirm nightly jobs succeed consistently.
- Keep artifact parsing and docs generation idempotent and tolerant to missing fields; include defensive validation and publish tests that guard the long-running job outputs.

## Acceptance Criteria (Definition of Done)
- `specs/001-benchmarks-specs/spec.md` exists and defines migration of doctests with concrete Given/When/Then and thresholds.
- A runner (`opt/benchmark/runner.py`) exists with CLI and programmatic API and has unit tests.
- `opt/history` stores compressed history artifacts validated by `schemas/benchmark-data-schema.json`.
- `tests/benchmark_quick` contains tests that run in <1s on GitHub-hosted runners and pass in PR checks.
- `.github/workflows/benchmark-quick.yml` is present and runs on PRs; `.github/workflows/benchmark-nightly.yml` scheduled and uploads artifacts.
- Docs generation pipeline can consume artifacts and produce draft docs pages under `docs/benchmarks/generated/` and a sample update is produced in the PR.
- The migration for `MantaRayForagingOptimization` (as a canonical example) is complete with a passing benchmark_quick test and validated artifact.

## Research Tasks (Phase 0 unresolved questions)
- Decide compression standard (gz vs zstd) and document trade-offs (speed vs compression ratio vs availability in GH Actions).
- Define retention policy for artifacts and whether to use external storage for long-term retention.
- Determine exact schema fields for `benchmark-data-schema.json` and what derived statistics to include (ERT, best_fitness history length, runtime/memory stats).
- Define CI budget for nightly runs and throttling/backoff strategies for flaky tests.

## Next Steps
1. Create `specs/001-benchmarks-specs/spec.md` and `research.md` (Phase 0) and iterate until all Research Tasks are resolved.
2. Proceed to Phase 1 implementation tasks once Constitution Check passes.
3. Schedule Phase 2 CI integration and GitHub validation after implementation and tests pass locally.



```
