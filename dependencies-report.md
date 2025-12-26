# Dependency Audit — quick summary

Date: 2025-12-26

Result: Static dependency audit found no vulnerable or deprecated packages in `pyproject.toml` at time of scan.

Recommendations:
- Continue periodic scans with `pip-audit` or `safety` in CI.
- Consider adding `pip-audit` to the CI validation pipeline (weekly job).
