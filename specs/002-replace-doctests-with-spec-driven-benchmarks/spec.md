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
- **FR-005**: Add GitHub Actions workflow with two jobs: `quick` (PR checks) and `full` (nightly), including artifact upload and `specify check` enforcement.- **FR-010**: Add explicit early stopping support to all optimizers and benchmark helpers: expose `early_stop` (or `stop_threshold`) parameter and ensure benchmark outputs include `iterations` (int), `stopped_early` (bool), and `stopping_reason` (str). Benchmarks used for quick PR checks MUST use conservative `max_iter` defaults (e.g., `<= 500`) and prefer early stopping; **do not use** `max_iter=10000` as a default in benchmark comparisons without documented justification.
- **FR-011**: Add an audit script and a pre-commit check that flags unexplained `max_iter>=5000` literals in code, tests, and docs; audit results must be included in the PR when changing default `max_iter` behavior.- **FR-006**: Update docs (optimizer docstrings, `docs/algorithms/*`, and README) with example outputs and instructions to validate locally.
- **FR-007**: Perform changes in a single, comprehensive PR targeting branch `feat/benchmarks-specs-ci` unless maintainers instruct otherwise.
- **FR-008**: Expose a `benchmark(..., store=True, out_path: str|None)` method on `AbstractOptimizer` and `AbstractMultiObjectiveOptimizer` that runs a short internal benchmark, returns `{'path': str, 'metadata': dict}` and, when `store=True`, writes validated JSON artifacts to `benchmarks/output/`.
- **FR-009**: Add `export_benchmark_json(data: dict, path: str, schema: str='docs/schemas/benchmark-data-schema.json')` helper to the codebase that writes, validates (using `jsonschema`), and returns the artifact path.

### Implementation Targets (mandatory)

- **IT-001**: `opt/abstract_optimizer.py` — add method:

```py
def benchmark(self, *, store: bool = True, out_path: str | None = None, schema: str = 'docs/schemas/benchmark-data-schema.json', **kwargs) -> dict:
    """Run a short, internally-configured benchmark and optionally store the result.

    Returns:
        dict: {"path": path_to_json, "metadata": {"seed": seed, "timestamp": iso_ts, "schema_version": ...}}
    """
```

- **IT-002**: `opt/multi_objective/abstract_multi_objective.py` — add equivalent `benchmark` method returning multi-objective-compatible metadata.

- **IT-003**: `opt/benchmark/utils.py` — add `export_benchmark_json(data, path, schema)` and `validate_benchmark_json(path, schema)` helper functions. Use `jsonschema` for validation and raise descriptive errors on failures.

- **IT-004**: `scripts/replace_trivial_doctests.py` — implement a script that finds trivial doctest patterns and replaces them with a mini-benchmark docstring example (requires manual review before commit). Provide a `--dry-run` mode and a `--apply` flag.
- **IT-005**: Add early stopping support to `AbstractOptimizer` and `AbstractMultiObjectiveOptimizer` (signature parameter `early_stop` / `stop_threshold`) and propagate this behavior to concrete optimizer implementations; ensure `benchmark(..., store=True)` captures `iterations`, `stopped_early`, and `stopping_reason` in metadata.
- **IT-006**: Implement `scripts/audit_max_iter.py` which scans the repo for occurrences of `max_iter` literals (e.g., `max_iter=10000`) and emits a report; add `pre-commit` hook `scripts/check_max_iter_precommit.py` to flag unexplained `max_iter>=5000` occurrences.
- **IT-007**: Add unit tests for early stopping behavior across representative optimizers and tests for the audit script (`tests/test_early_stop.py`, `tests/test_audit_max_iter.py`).

- **IT-005**: Tests:

  - `tests/test_benchmarks_quick.py`: parameterized quick tests, including in-spec examples.
  - `tests/test_benchmark_export.py`: tests for `export_benchmark_json` and `validate_benchmark_json`.
  - `tests/test_doctest_replacements.py`: asserts that the replacement script produces syntactically valid mini-benchmark examples.

- **IT-006**: CI & Pre-commit:
  - Add `.github/workflows/benchmarks.yml` with `quick` job (PRs) and `full` job (nightly) with artifact upload to `benchmarks/output/`.
  - Add `validate-benchmark-json` pre-commit hook to call schema validation on changed benchmark artifacts.

### Tools & MCPs (required for spec generation and PRs)

- `mcp_context7_get-library-docs` (fetch Spec Kit and COCO docs)
- `mcp_context7_resolve-libary-id` (resolve useful-optimizer repo)
- `mcp_ai-agent-guid_prompt-chaining-builder` (generate the stepwise spec)
- `mcp_ai-agent-guid_documentation-generator-prompt-builder` (produce doc examples)
- `mcp_ai-agent-guid_guidelines-validator` (validate docstring & schema rules)
- `mcp_ai-agent-guid_gap-frameworks-analyzers` (produce short gap-analysis for PRs)
- `mcp_serena_*` (activate/find/insert/replace for symbol-level edits)
- GitHub MCPs used to create branches and PRs

### Non-Functional Requirements

- **NFR-001**: Quick tests: target runtime <1s on typical dev hardware
- **NFR-002**: Full runs: artifacts compressed and retained at least 30 days
- **NFR-003**: History artifact compressed size per full-run ≤ 200MB (guideline)
- **NFR-004**: All tests must be deterministic with fixed seeds (reproducible within tolerance: use `np.allclose` for float comparisons with atol=1e-12 when checking reproducibility)
- **NFR-005**: Benchmarks and tests MUST not rely on excessive `max_iter` defaults for correctness; set reasonable quick-test defaults and document any exceptions. Pre-commit checks must flag `max_iter>=5000` to avoid accidentally shipping very long budgets in PRs.

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

### In-spec minimal example (sanity test)

Include the following exact in-spec example in docstrings or spec examples. This snippet must execute as a quick sanity check and validate the generated JSON artifact against the schema.

```py
>>> from opt.benchmark.functions import sphere
>>> opt = SimulatedAnnealing(func=sphere, lower_bound=-5, upper_bound=5, dim=2, max_iter=100, early_stop=1e-6, seed=42)
>>> result = opt.benchmark(store=True)
>>> import jsonschema, json
>>> with open(result['path']) as f: data = json.load(f)
>>> jsonschema.validate(instance=data, schema_path='docs/schemas/benchmark-data-schema.json')  # should not raise
>>> assert data['metadata']['iterations'] <= 100
>>> assert isinstance(data['metadata'].get('stopped_early'), bool)
True
```

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
- Add `tests/test_early_stop.py` verifying `early_stop` stops before `max_iter` when threshold met and sets `metadata['stopped_early'] == True` and `metadata['iterations']` accurately.
- Add `tests/test_audit_max_iter.py` to validate `scripts/audit_max_iter.py` finds large `max_iter` literals and that the pre-commit check flags them when appropriate.

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

## Project Plan & Review Cadence

- **Estimate & milestones (recommended):**
  - Sprint 1 (2 working days): Add `benchmark` method to `AbstractOptimizer` and add `export_benchmark_json` helper; add unit tests for export/validation.
  - Sprint 2 (3 working days): Implement `scripts/replace_trivial_doctests.py` and add `tests/test_doctest_replacements.py`; run replacements in dry-run mode and review manually.
  - Sprint 3 (2 working days): Add `benchmark_quick` tests, CI `quick` job, and pre-commit `validate-benchmark-json` hook; run quick suite on PRs.
  - Stabilization (1 week): Tune full-run thresholds, add nightly `benchmark_full` workflow with artifact upload, and perform first full-nightly run to tune thresholds and archive artifacts.
- **Review cadence:**
  - PR reviewers must run `benchmark_quick` locally or via CI job; PRs require gap-analysis via `mcp_ai-agent-guid_gap-frameworks-analyzers` and a Constitution Check.
  - Hold short retrospective after the first full-nightly run to update thresholds and fix flaky tests.
- **Feedback loops & iteration:**
  - Continuous monitoring of nightly artifacts, triage issues, and apply fixes in subsequent sprints.
  - Use `specify check` and the Spec Kit gates in planning to validate readiness before merging.
  - **Agile practices:** adopt a Scrum-like cadence: 2-week sprints, daily standups, sprint planning, sprint review, and a retrospective after each sprint to adjust estimates and priorities. Use retrospective outcomes to update acceptance thresholds and velocity estimates.
- **Estimation methodology & capacity assumptions:**
  - Estimates above assume one engineer at ~80% capacity (time for meetings and code review included) and conservative story-point conversion (1 point ≈ 1 day of focused work).
  - Expected total lead time for the work outlined: **2–3 weeks** (including stabilization and first nightly runs). Adjust after first sprint using observed velocity.

### Timeline summary (data-driven estimates)

| Sprint        | Duration | Primary tasks                                                   | Story points (est.) |
| ------------- | -------- | --------------------------------------------------------------- | ------------------- |
| Sprint 1      | 2 days   | `benchmark` method + `export_benchmark_json` helper             | 4                   |
| Sprint 2      | 3 days   | replacement script + doctest replacement review + tests         | 6                   |
| Sprint 3      | 2 days   | add `benchmark_quick` tests, CI `quick` job, pre-commit hooks   | 4                   |
| Stabilization | 5 days   | nightly `benchmark_full` job, artifact upload, threshold tuning | 10                  |

_Total_: ~17 working days (adjustable with velocity observed after first sprint).

## Notes

- Replace doctests programmatically when possible (script to search/replace trivial patterns) but each replacement must be manually reviewed for semantic correctness.
- Keep acceptance thresholds conservative in full-run tests to avoid flaky failures; tune thresholds after first full-nightly run.

---

**Spec author**: GitHub Copilot (via contributor)
**Ready for**: `/speckit.clarify` (if any ambiguous thresholds remain) and then `/speckit.plan` to create an implementation plan.
