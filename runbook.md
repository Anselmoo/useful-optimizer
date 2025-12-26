# Runbook — Benchmarks nightly job

## Smoke tests
- `uv run pytest -q -k benchmark_quick` (should pass)
- Validate latest artifact: `python scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json benchmarks/output/latest.json`

## Rollback
- Disable nightly GitHub Actions workflow: rename `.github/workflows/benchmarks-full.yml` to `.github/workflows/benchmarks-full.disabled.yml` and push to main.
- Re-run last-known-good nightly: download artifact from run X and re-run validation locally.

## Contacts
- Maintainers: @AnselmHahn
- Benchmarking SME: @benchmark-expert
