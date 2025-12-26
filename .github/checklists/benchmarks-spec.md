# Benchmarks Spec Checklist

Purpose: Reviewer checklist to verify that trivial doctests were replaced with spec-driven mini-benchmarks, tests and CI enforce schema-valid benchmark artifacts, and archival runs are configured.

How to use locally: run the commands in the Acceptance column and ensure they pass; annotate the PR with this checklist and attach resulting artifacts.

## Core checks

- [ ] CHK001 - Spec exists for each replaced doctest (`grep -q "benchmark(store=True)" opt -R`) [Completeness]
  - Local command: grep -R "benchmark(store=True)" opt/ || echo "No benchmark examples found"

- [ ] CHK002 - `benchmark_quick` tests pass (PR quick checks) [Acceptance]
  - Local command: uv run pytest -q -k benchmark_quick

- [ ] CHK003 - Generated JSON artifacts validate against schema [Acceptance]
  - Local command: python scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json benchmarks/output/*.json

- [ ] CHK004 - Pre-commit hook `validate-benchmark-json` runs and passes on generated artifacts [Process]
  - Local command: pre-commit run validate-benchmark-json --files benchmarks/output/*.json

- [ ] CHK005 - CI quick job exists and runs `uv run pytest -q -k benchmark_quick` on PRs [CI]
  - Verify: `.github/workflows/benchmarks-quick.yml` exists and your PR shows a passing job in Actions

- [ ] CHK006 - CI nightly job exists and uploads artifacts to `benchmarks/output/` and archives them [CI]
  - Verify: `.github/workflows/benchmarks-full.yml` exists and retains artifacts for archival/inspection

- [ ] CHK007 - Benchmark artifacts follow naming & metadata conventions [Artifacts]
  - Pattern: `benchmarks/output/<algorithm>-<YYYYMMDDTHHMMSSZ>-s<seed>.json` and include `schema_version` metadata
  - Local: ls -1 benchmarks/output | head -n 5

- [ ] CHK008 - `track_history` remains opt-in and large histories are not stored by default [Non-functional]
  - Review code changes that add history and ensure `track_history` defaults to `False` or is opt-in in examples

- [ ] CHK009 - Forbidden MCP patterns are not present in runtime code [Security]
  - Local command: python scripts/check_forbidden_mcp_calls.py opt/ || true

- [ ] CHK010 - PR includes a short gap-analysis (3-5 bullets) generated via `mcp_ai-agent-guid_gap-frameworks-analyzers` and addressed in PR description [Review Criteria]
  - Check: PR description includes "Gap Analysis:" section and references `.github/analysis/benchmarks-gap-report.md` if applicable

## Extra checks (recommended)

- [ ] CHK011 - Tests are deterministic with fixed seed and reproducible within tolerance (`np.allclose` with `atol=1e-12`) [Measurability]
  - Example test assertion: assert np.allclose(sol1, sol2, atol=1e-12)

- [ ] CHK012 - Docs updated: docstrings and `docs/` contain updated mini-benchmark examples and a short "How to validate locally" section [Docs]
  - Local: grep -R "benchmark(store=True)" docs/ && grep -R "How to validate" docs/ || echo 'Docs need updates.'

- [ ] CHK013 - Pre-commit & CI enforce forbidden pattern checks and schema validation [Process]
  - Verify pre-commit config includes `validate-benchmark-json` and `check_forbidden_mcp_calls` hooks

- [ ] CHK014 - Runbook and rollback steps included for nightly archival job [Release]
  - Files: `runbook.md`, `rollback-commands.md`, `ci/archiving.md` referenced in PR

## CI smoke job suggestion (optional)
Add a small job to `ci/smoke.yml` for PRs that runs the checklist commands (non-invasive checks):

```yaml
name: Benchmarks Checklist Smoke
on: [pull_request]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: uv sync
      - name: Run quick checks
        run: |
          uv run pytest -q -k benchmark_quick
          uv run python scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json benchmarks/output/*.json || true
```

## Notes
- This checklist is intended for reviewers to **test the SPEC**, not implementations. Focus on presence, clarity, measurability, and traceability of requirements and artifacts.
- Use `mcp_serena_insert_after_symbol` to automatically inject checklist references into PR templates or contributing docs when appropriate.

---

Generated via: `mcp_ai-agent-guid_quick-developer-prompts-builder` + validated with `mcp_ai-agent-guid_guidelines-validator`.
