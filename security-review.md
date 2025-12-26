# Security & Hardening Review — Benchmarks feature

Date: 2025-12-26

Executive Summary:
- Risk: Artifacts may accidentally include sensitive data or large binary content; pre-commit and CI controls mitigate risk.
- Recommendation: Implement `validate-benchmark-json` pre-commit hook, file size checks, filename sanitation, and forbidden-pattern checks (e.g., `mcp_` in runtime `opt/` files).

Actions:
- Add pre-commit hook `scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json`.
- Add `scripts/check_forbidden_mcp_calls.py` to detect `mcp_` patterns and fail pre-commit if found.
- Enforce artifact size limit in CI (e.g., >200MB fail with error and suggest compress/trim history).
- Validate artifact content for secret patterns (use a simple grep of PEM/private key headers and common secrets) as a pre-commit/CI step.

Test cases:
- Attempt to commit a JSON with a PEM private key: pre-commit hook fails and explains reason.
- Commit a large artifact (>200MB): pre-commit fails (or CI fails) and instructs to compress or opt out by justification.
