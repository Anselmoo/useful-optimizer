# Contributing

## Adding an optimizer

1. Create `src/opt/<category>/<file_stem>.py` with one class that inherits from
   `AbstractOptimizer` (see [Abstract Optimizer](/api/abstract-optimizer)) and export it from the
   category's `__init__.py` and from `opt/__init__.py`.
2. Write the class docstring in Google style with the sections the validator expects:
   *Algorithm Metadata* (name, acronym, year, authors), *Mathematical Formulation*,
   *Hyperparameters*, *COCO/BBOB Benchmark Settings*, *Args*, *Attributes*, *Example*,
   *References* and *See Also*. Look at `whale_optimization_algorithm.py` for a complete one.
3. Record history with `self._record_history(...)` in the loop and call
   `self._finalize_history()` before `search()` returns.
4. Regenerate the documentation and commit the result:

   ```bash
   uv run python scripts/generate_docs.py --all --json --sidebar
   uvx --from repo-release-tools rrt artifacts --snapshot
   ```

## Naming rules

| Thing | Rule | Example |
|-------|------|---------|
| Docs URL | file stem with `_` as `-` | `whale_optimization_algorithm.py` → `/algorithms/swarm-intelligence/whale-optimization-algorithm` |
| Page title | `Algorithm Name` in the metadata table | *Whale Optimization Algorithm* |
| Acronym | `Acronym` in the metadata table, unique across the package | `WOA` |

The page, JSON entry and sidebar link of every optimizer come from these three rules, and
`scripts/test/test_docs_consistency.py` fails if they drift apart.

## Checks

```bash
uv run pytest src/opt/test/ scripts/test/        # unit tests and docs consistency
uv run pytest --doctest-modules src/opt          # doctests
uv run ruff check src/opt/ && uv run ruff format --check src/opt/
cd docs && npm ci && npm run docs:build          # fails on dead links
```

Commit subjects follow [Conventional Commits](https://www.conventionalcommits.org/); the release
policy workflow checks them, the generated-file fingerprints and the version targets.
