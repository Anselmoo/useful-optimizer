# Feature Specification: Replace trivial doctests with spec-driven benchmark tests

**Feature Branch**: `002-replace-doctests-with-spec-driven-benchmarks`
**Created**: 2025-12-25
**Status**: Draft
**Input**: "Replace trivial doctest examples (for example, `len(solution) == dim`) with Spec Kit feature specs and test harness that provide reproducible benchmarks, fast sanity checks for PR feedback, and nightly full-run validation with history artifacts conforming to repository schemas."

## User Scenarios & Testing _(mandatory)_

### User Story 1 — Replace trivial doctests in optimizer docstrings (Priority: P1)

A developer or reviewer must be able to read an optimizer docstring example and see a reproducible, meaningful verification that the optimizer ran and produced verifiable, deterministic output.

**Why this priority**: Prevents false confidence from trivial doctests and provides reproducible examples that serve as both documentation and lightweight checks.

**Independent Test**: Run automated script to find former trivial doctests and verify that each replaced example can be executed and passes the `benchmark_quick` test suite.

**Acceptance Scenarios**:

1. **Given** an optimizer module with a trivial doctest example, **When** the spec is applied, **Then** the docstring contains a seeded usage example and a deterministic assertion (e.g., reproducibility check and basic sanity assertions) and an associated `benchmark_quick` test exists.
2. **Given** running the module's quick test locally with `uv run pytest -q -k benchmark_quick`, **When** executed, **Then** the new tests complete in <1s and pass on a typical development machine.

---

### User Story 2 — Add fast sanity tests for PR feedback (Priority: P1)

PRs should execute `benchmark_quick` tests (seeded, deterministic) and fail if regressions are detected.

**Independent Test**: On PRs, the quick job runs and passes before merge.

**Acceptance Scenarios**:

1. **Given** a quick sanity test exists for an optimizer, **When** run against current HEAD, **Then** it finishes in <1s and verifies shape, deterministic reproducibility with fixed seed, and that history entries exist if `track_history=True`.

---

### User Story 3 — Add nightly full-run benchmarks with artifact upload (Priority: P2)

Scheduled CI jobs must run `@pytest.mark.benchmark_full` tests nightly, upload compressed history artifacts conforming to `docs/schemas/benchmark-data-schema.json`, and surface regressions as CI failures.

**Independent Test**: The scheduled job runs successfully, artifacts are stored, and a single run can be reconstituted into the docs' benchmark comparison pipeline.

**Acceptance Scenarios**:

1. **Given** the nightly job, **When** it finishes, **Then** compressed history JSON artifacts are attached to workflow run and validated against `docs/schemas/benchmark-data-schema.json`.
2. **Given** a regression in the acceptance metrics (per-spec thresholds), **When** the nightly job detects it, **Then** an issue is opened/flagged for triage and included in the next sprint backlog.

---

### Edge Cases

- Runner non-determinism (external RNGs, parallel reductions): specify how to control (np.random.default_rng with seed; disable parallel non-deterministic pathways in quick tests).
- Algorithms that early-stop or raise for specific params: tests should capture and assert expected exception classes or documented behavior.
- Dimension mismatches: verify doc example `dim` aligns with `len(solution)` and the assertion checks shape, not just length equality.
- Very small budgets: quick tests intentionally use small `max_iter` (e.g., 50) with conservative assertions (reproducibility + history presence) and avoid strict numeric thresholds unless in full-run tests.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: Replace trivial doctest examples in all optimizer docstrings with seeded, reproducible usage examples that include a deterministic check and a short sanity assertion.
- **FR-002**: Add `benchmark_quick` pytest tests for each optimizer example (fast, deterministic, <1s), using `seed` and `track_history` where applicable.
- **FR-003**: Add `@pytest.mark.benchmark_full` integration tests for nightly CI runs that run longer budgets and assert numeric performance goals (e.g., fitness < threshold for sphere with conservative thresholds).
- **FR-004**: Validate that history artifacts are produced in memory-efficient format and conform to `docs/schemas/benchmark-data-schema.json`.
- **FR-005**: Add GitHub Actions workflow with two jobs: `quick` (PR checks) and `full` (nightly), including artifact upload and `specify check` enforcement.
- **FR-006**: Update docs (optimizer docstrings, `docs/algorithms/*`, and README) with example outputs and instructions to validate locally.
- **FR-007**: Perform changes in a single, comprehensive PR targeting branch `feat/benchmarks-specs-ci` unless maintainers instruct otherwise.

### Non-Functional Requirements

- **NFR-001**: Quick tests: target runtime <1s on typical dev hardware
- **NFR-002**: Full runs: artifacts compressed and retained at least 30 days
- **NFR-003**: History artifact compressed size per full-run ≤ 200MB (guideline)
- **NFR-004**: All tests must be deterministic with fixed seeds (reproducible within tolerance: use `np.allclose` for float comparisons with atol=1e-12 when checking reproducibility)

## Key Entities

- **Optimizer**: algorithm class, e.g., `SequentialMonteCarloOptimizer`
- **BenchmarkRun**: an execution with params (seed, dim, bounds, max_iter, population_size)
- **HistoryArtifact**: compressed JSON following `docs/schemas/benchmark-data-schema.json`
- **Spec**: Spec Kit feature specification file and related checklists

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: All trivial doctests are replaced in `opt/` modules and validated by `benchmark_quick` tests.
- **SC-002**: `benchmark_quick` suite completes in <1s on CI PR runners and locally on typical dev machines.
- **SC-003**: Reproducibility: running a `benchmark_quick` test twice with the same seed yields identical results (solutions and fitness equal within `np.allclose(..., atol=1e-12)`).
- **SC-004**: Nightly full-run artifacts are uploaded and successfully validate against `docs/schemas/benchmark-data-schema.json`.
- **SC-005**: Single PR merges that include the spec(s), tests, CI changes, and documentation and closes issue #85 and fully integrates PR #119 changes.

## How to validate locally _(mandatory)_

1. Sync environment:

```fish
uv sync
```

2. Run fast sanity tests (for a single optimizer locally):

```fish
uv run pytest tests/ -q -k benchmark_quick -q
```

3. Run a single full benchmark test locally (example):

```fish
uv run pytest tests/ -q -m benchmark_full -k test_some_optimizer_full
```

4. Run Spec Kit compliance:

```bash
specify init . --force
specify check
```

5. Validate a sample history artifact against schema:

```python
python -c "import json, jsonschema, gzip; j=json.load(gzip.open('path/to/history.json.gz')); jsonschema.validate(j, json.load(open('docs/schemas/benchmark-data-schema.json')))"
```

6. Lint & format

```fish
uv run ruff check opt/ && uv run ruff format opt/
```

## Testing Plan _(mandatory)_

- Add `tests/test_benchmarks_quick.py` with parameterized quick tests for a representative set of optimizers (seeded runs, shape check, reproducibility check, history minimality check).

**Quick-test pattern (example)**

```python
def test_optimizer_quick_example_sanity():
    optimizer = SequentialMonteCarloOptimizer(func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=50, seed=42, track_history=True)
    solution, fitness = optimizer.search()
    # Basic structure checks
    assert solution.shape == (10,)
    assert isinstance(fitness, float)
    # Reproducibility
    s2, f2 = SequentialMonteCarloOptimizer(func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=50, seed=42, track_history=True).search()
    assert np.allclose(solution, s2, atol=1e-12)
    assert np.allclose(fitness, f2, atol=1e-12)
    # History minimality
    assert 'best_fitness' in optimizer.history
```

**Full-test pattern (nightly)**

```python
@pytest.mark.benchmark_full
def test_optimizer_full_sphere():
    optimizer = SequentialMonteCarloOptimizer(func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=2000, seed=0, track_history=True)
    solution, fitness = optimizer.search()
    assert fitness < 1.0  # Conservative full-run threshold for sphere dim=10
    # Save and validate history artifact against schema
```

## Dependencies & Constraints

- COCO/BBOB compliance: follow guidelines in `docs/.` and use seeds 0-14 for statistical runs if used in comparison experiments.
- Use `np.random.default_rng(self.seed)` for RNG in all optimizers to ensure consistent seeding across implementations.
- Ensure `track_history` implementations use memory-efficient structures (PR #119 approach recommended).

## Notes

- Replace doctests programmatically when possible (script to search/replace trivial patterns) but each replacement must be manually reviewed for semantic correctness.
- Keep acceptance thresholds conservative in full-run tests to avoid flaky failures; tune thresholds after first full-nightly run.

---

**Spec author**: GitHub Copilot (via contributor)
**Ready for**: `/speckit.clarify` (if any ambiguous thresholds remain) and then `/speckit.plan` to create an implementation plan.
