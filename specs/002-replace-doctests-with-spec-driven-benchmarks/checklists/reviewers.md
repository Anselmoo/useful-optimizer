# Reviewers' Checklist: Replace trivial doctests with spec-driven benchmark tests

**Purpose**: Ensure pull requests implementing this feature meet Spec Kit, testing, reproducibility, observability, and CI standards.

**Scope**: Applies to PR `feat/benchmarks-specs-ci` and all follow-ups that replace doctests, add tests, CI, and history validation.

## Quick Commands (local validation)

- Run quick tests: `uv run pytest -q -k benchmark_quick`
- Run full tests (locally): `uv run pytest -q -m benchmark_full`
- Run docstring/spec checks: `specify init . --force && specify check`
- Run trivial doctest checker: `python scripts/check_trivial_doctests.py` (should exit 0)
- Validate history artifact: `python -c "import json, jsonschema, gzip; j=json.load(gzip.open('path/to/history.json.gz')); jsonschema.validate(j, json.load(open('docs/schemas/benchmark-data-schema.json')))"`

---

## Review Checklist Items

Each item should be checked during PR review and listed in the PR description. Prefix checks with `[ ]` or `[x]` as appropriate.

### A. Spec & Documentation [Completeness & Clarity]

- [ ] **Spec exists** for every replaced doctest: Spec file in `specs/` referencing the changed module(s) [Spec §FR-001]
- [ ] **Given/When/Then** scenarios present with measurable acceptance criteria (including numeric thresholds and reproducibility) [Spec §User Stories]
- [ ] **'How to validate locally'** steps present in spec and PR description
- [ ] **Docs replaced**: Docstring examples replaced; `docs/algorithms/*` and README reflect changes

### B. Tests [Coverage & Determinism]

- [ ] `benchmark_quick` tests are implemented and parameterized for representative optimizers
- [ ] Quick tests run in <1s (verify locally and in CI) and pass on PR
- [ ] Reproducibility checks included (fixed seeds; `np.allclose` tolerances documented) [Spec §NFR-004]
- [ ] `@pytest.mark.benchmark_full` tests present and marked for nightly runs (not part of PR quick job)
- [ ] Tests verify `track_history=True` behavior where applicable and assert minimal history structure

### C. Code Quality & Patterns

- [ ] RNG usage is consistent (`np.random.default_rng(self.seed)` where applicable)
- [ ] History storage follows memory-efficient pattern (PR #119 approach) and code includes comments about retention/size
- [ ] No remaining trivial doctest patterns (`scripts/check_trivial_doctests.py` passes)
- [ ] Linting/formatting passes (`uv run ruff check opt/ && uv run ruff format opt/`) and pre-commit hooks run cleanly

### D. CI & Automation

- [ ] `specify check` runs in PR job and passes
- [ ] `quick` GitHub Actions job exists and runs `benchmark_quick` tests, linter, and `specify check`
- [ ] `full` scheduled GitHub Actions job exists (nightly) and runs `@pytest.mark.benchmark_full` tests
- [ ] Nightly job uploads compressed history artifacts (`.json.gz`) and runs schema validation against `docs/schemas/benchmark-data-schema.json`
- [ ] Nightly job failure on regression creates an issue or alert per governance rules

### E. Artifacts & Observability

- [ ] History artifacts conform to schema and are compressed (use JSON schema validation in test)
- [ ] Artifact sizes are within guidance (≤ 200MB compressed) or exceptions documented in plan
- [ ] Artifacts include seed, environment (Python + deps), run parameters, and minimal summary metrics (best fitness, runtime, evals)

### F. PR Hygiene & Governance

- [ ] PR includes link to spec(s) and checklist items
- [ ] Conventional commit message used (e.g., `feat: replace trivial doctests with spec-driven tests + CI`)
- [ ] At least one maintainer review requested and CI green before merge
- [ ] Merge closes issue #85 and references PR #119 integration status

---

## Triage Guidance for Failures

- Quick-test failure on PR: fix immediately (tests must be deterministic); if flakiness suspected, mark the test and add to follow-up with `@pytest.mark.flaky` + issue to stabilize.
- Nightly full-run regression: workflow must open an issue with artifact links and assign to maintainers; triage within 48 hours per Constitution.

---

**Notes**:

- Keep acceptance criteria conservative at first (avoid brittle thresholds until nightly data is collected).
- For performance-sensitive changes, include a short benchmark note in the PR describing why thresholds were chosen.
- If you want, run `python scripts/check_trivial_doctests.py` locally and fix found examples, then add spec/test and push in the single PR branch `feat/benchmarks-specs-ci`.
