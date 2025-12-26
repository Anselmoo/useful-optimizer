# Tasks — Replace trivial doctests with spec-driven benchmarks

This file has been consolidated. The canonical task list is maintained at:

`specs/002-replace-doctests-with-spec-driven-benchmarks/tasks.md`

Please make edits there. This root-level file exists only to point to the canonical spec tasks.

Canonical tasks path: `specs/002-replace-doctests-with-spec-driven-benchmarks/tasks.md`

## Snapshot: TIDs 001-011

- T001: Implement `export_benchmark_json` & `validate_benchmark_json` — Owner: @you — Acceptance: `pytest tests/test_benchmark_export.py`
- T002: Add `benchmark()` to `AbstractOptimizer` — Owner: @you — Acceptance: `uv run pytest -q -k benchmark_quick`
- T003: Implement `scripts/replace_trivial_doctests.py` (dry-run & apply) — Owner: @you — Acceptance: `tests/test_doctest_replacements.py`
- T004: Add CI quick job (`benchmarks-quick.yml`) — Owner: @you — Acceptance: CI quick job appears and passes
- T005: Add nightly `benchmark_full` workflow — Owner: @you — Acceptance: nightly artifacts uploaded and validated
- T006: Update docs & docstrings — Owner: @docs — Acceptance: `grep -R "benchmark(store=True)" docs/`
- T007: Add `validate-benchmark-json` pre-commit hook — Owner: @you — Acceptance: pre-commit hook runs locally
- T008: Add forbidden MCP check in pre-commit — Owner: @you — Acceptance: `python scripts/check_forbidden_mcp_calls.py opt/` exits non-zero on forbidden patterns
- T009: Add `benchmark_quick` tests — Owner: @you — Acceptance: quick suite completes <1s and passes
- T010: Add runbook & archiving docs — Owner: @you — Acceptance: runbook contains smoke test commands
- T011: Add CI smoke job — Owner: @you — Acceptance: smoke job runs and reports pass/fail (non-blocking)
