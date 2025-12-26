# Discovery: Replace trivial doctests with spec-driven benchmarks

**Generated**: 2025-12-26
**Branch**: `001-benchmarks-specs-ci`

Summary

- We scanned the repository for long `max_iter` literals (e.g., `max_iter=10000`) and occurrences of `early_stop` usage.
- **Findings**:
  - Multiple docstrings and examples across `opt/` use `max_iter=10000` (see `discovery/files.json` for details).
  - Early stopping is implemented in some optimizers (e.g., `opt/social_inspired/social_group_optimizer.py`) but **not uniformly** across the codebase.
  - No global pre-commit audit script exists yet to flag large `max_iter` literals.

Files of interest (see `files.json` for full list)

- `opt/evolutionary/cma_es.py`
- `opt/evolutionary/genetic_algorithm.py`
- `opt/evolutionary/differential_evolution.py`
- `opt/probabilistic/sequential_monte_carlo.py`
- `scripts/batch_update_docstrings.py`
- `.github/prompts/optimizer-docs-template.prompt.md`

Next actions (discovery → design handoff)

1. Implement `scripts/audit_max_iter.py` to produce a report of `max_iter` occurrences and their context (file, line, snippet). Output file: `reports/max_iter_audit.json`.
2. Implement pre-commit script `scripts/check_max_iter_precommit.py` that checks staged files for `max_iter>=5000` and fails pre-commit if no justification comment is present.
3. Add `early_stop` support to `AbstractOptimizer` and `AbstractMultiObjectiveOptimizer` and write migration guide for implementing consistent early-stop behavior across optimizers.
4. Prioritize changing docstring examples to `max_iter=500` with `early_stop` usage for quick benchmarks; leave full-run examples with documented justification where necessary.

Recommended MCP calls to validate decisions and collect best practices:

- `mcp_ai-agent-guid_code-analysis-prompt-builder({analysisType: 'performance', content: 'scan for missing early-stop and inefficient loop bounds like max_iter=10000', codebase: '/Users/hahn/LocalDocuments/GitHub_Forks/useful-optimizer'})`
- `vscode-websearchforcopilot_webSearch({query: 'early stopping thresholds optimizer max_iter best practices'})`
- `mcp_ai-agent-guid_gap-frameworks-analyzers({frameworks: ['testing','performance','process'], currentState: 'repo snapshot', desiredState: 'spec-driven benchmarks'})`

MCP logging

- All the above recommended MCP calls should be recorded to Serena memory with `mcp_serena_write_memory` and referenced in `plan.md` and generated artifacts for auditability.

Discovery artifacts

- `specs/002-replace-doctests-with-spec-driven-benchmarks/discovery/files.json` (machine-readable file index)
- This file: `specs/002-replace-doctests-with-spec-driven-benchmarks/discovery/discovery.md`
