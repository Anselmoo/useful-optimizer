# CI Archiving & Retention — Benchmarks artifacts

Policy:
- Nightly CI uploads artifacts to `benchmarks/output/` and the workflow attaches them to the workflow run.
- Files must follow naming convention: `benchmarks/output/<algorithm>-<YYYYMMDDTHHMMSSZ>-s<seed>.json` and include `schema_version` in metadata.
- Retention: keep nightly artifacts for 30 days, then rotate to long-term storage (S3 or an artifact bucket) per infra policy.
- Validation: Any uploaded artifact must pass `scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json` before final archival.

Suggested GitHub Actions steps:
- `upload-artifact` after validation
- `persist-to-s3` or external bucket in a separate job running on successful validation
