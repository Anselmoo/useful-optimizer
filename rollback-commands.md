# Rollback commands

To quickly disable the nightly full-run workflow:

```bash
# On local branch
git checkout main
# Rename workflow to disable
git mv .github/workflows/benchmarks-full.yml .github/workflows/benchmarks-full.disabled.yml
git add -A && git commit -m "chore(ci): disable nightly benchmarks full-run (rollback)" && git push origin main
```

To re-run last-known-good artifacts:

```bash
# Download artifact from workflow run page (or use gh cli)
gh run download <run-id> --name benchmarks-output
python scripts/validate_benchmark_json.py --schema docs/schemas/benchmark-data-schema.json benchmarks/output/*.json
```
