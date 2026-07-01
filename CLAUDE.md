# useful-optimizer — Claude Code Guide

## Project Overview

Python library of 120+ optimization algorithms (PSO, DE, SGO, etc.) with a VitePress documentation site featuring scientific benchmark visualizations. Package name: `useful-optimizer`, import root: `opt`.

## Tooling

- **Package manager**: `uv` (not pip/poetry)
- **Python**: `>=3.12,<3.15` (CI matrix: 3.12, 3.13, 3.14)
- **Linter/formatter**: `ruff` — runs on pre-commit; auto-fixes on commit
- **Type checker**: `ty` (`uv run ty check src/opt/`)
- **Tests**: `pytest` with `--doctest-modules` (doctests in every optimizer file are part of the test suite)
- **Docs**: VitePress 1.5 in `docs/`, Node 24+

## Common Commands

```bash
# Python
uv sync --all-extras          # install all deps including dev/benchmark/validate
uv run pytest src/opt/test/ -v --tb=short   # unit tests
uv run pytest --doctest-modules -v          # doctests
uv run ruff check src/opt/   # lint
uv run ruff format src/opt/  # format
uv run ty check src/opt/     # type check (non-blocking)

# Docs
cd docs && npm run docs:dev    # dev server
cd docs && npm run docs:build  # production build (runs SSR — must be SSR-safe)
cd docs && npm run docs:api    # regenerate Griffe JSON API files

# Full docs CI pipeline locally (mirrors docs.yaml build job):
uv run python scripts/generate_docs.py --all --json --griffe --full-api --sidebar
cd docs && npm ci && npm run docs:build && npm run validate:migration

# Benchmarks
uv run python benchmarks/run_benchmark_suite.py --output-dir benchmarks/output
```

## Repository Structure

```
src/opt/                    # Python package (120+ optimizers) — src layout since #144
  abstract/                 # AbstractOptimizer base class
  swarm_intelligence/       # 56 files
  evolutionary/             # 6 files
  gradient_based/           # 11 files
  classical/                # 9 files
  metaheuristic/            # 15 files
  physics_inspired/         # 4 files
  probabilistic/            # 5 files
  social_inspired/          # 4 files
  constrained/              # 5 files
  multi_objective/          # 3 files
  benchmark/                # benchmark functions (shifted_ackley, rosenbrock, sphere…)

benchmarks/                 # CI benchmark pipeline
  run_benchmark_suite.py    # main runner (produces benchmark-results.json)
  generate_plots.py         # matplotlib plots from results
  aggregate_results.py      # aggregation utilities

docs/                       # VitePress documentation site
  .vitepress/
    config.ts               # VitePress config + Catppuccin Mocha theme
    loaders/api.data.ts     # VitePress data loader: Griffe JSON → typed API
    theme/
      index.ts              # Theme entry — see SSR note below
      components/           # ConvergenceChart, ECDFChart, ViolinPlot, FitnessLandscape3D, APIDoc
      types/benchmark.ts    # TypeScript interfaces for benchmark JSON schema
      utils/benchmarkTransforms.ts  # Transform benchmark JSON → ECharts series
    types/griffe.d.ts        # TypeScript types for Griffe AST output
  api/                      # Griffe-generated JSON (*.json, gitignored output)
  public/
    benchmarks/             # Benchmark JSON served to the browser
      demo-benchmark-data.json
    optimizers/optimizers.json
  algorithms/               # Per-algorithm markdown pages
  benchmarks/               # Benchmark methodology docs
  guide/                    # User guide

scripts/                    # Dev tooling scripts
  generate_docs.py          # Griffe + sidebar generation
  unified_validator.py      # Pydantic schema validation for docstrings

.github/workflows/
  tests.yaml                # pytest matrix 3.12/3.13/3.14 + ruff lint (PRs + main)
  docs.yaml                 # VitePress build + Pages deploy (PRs + main; deploy skipped on PRs)
  docs-validation.yml       # Pydantic docstring schema validation
  benchmark-pipeline.yml    # COCO/BBOB CI pipeline (schedule/dispatch only)
  benchmark-visualizations.yaml  # Weekly benchmark run + artifact upload
```

## Pre-commit Hooks

Runs automatically on `git commit`:
1. `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`
2. `ruff` — lint with auto-fix
3. `ruff-format` — format (modifies files; re-stage and commit again if it fires)
4. `pydocstyle` — Google-style docstring compliance on `opt/` categories
5. `google-docstring-inline-summaries` — custom script enforcing inline summaries
6. `validate-optimizer-docstrings` — Pydantic schema validation

**When ruff-format fires and fails a commit:** re-stage the modified files and commit again — it auto-fixed them.

## Python Conventions

- All optimizer files must have Google-style docstrings with COCO/BBOB-compliant examples
- Doctests must not use `np.True_` — wrap boolean assertions in `bool()`
- Use `from __future__ import annotations` (enforced by ruff isort)
- Ruff `select = ["ALL"]` with specific ignores — see `pyproject.toml`
- History tracking: optimizers store `self.history = {"best_fitness": [], "mean_fitness": []}` updated each iteration

## VitePress / TypeScript Conventions

### SSR Safety (critical)
ECharts accesses `document` at module load time — it CANNOT be statically imported in `theme/index.ts`.
Chart components (ConvergenceChart, ECDFChart, ViolinPlot, FitnessLandscape3D) **must** be registered via `defineAsyncComponent` inside a `typeof window !== 'undefined'` guard:

```typescript
// theme/index.ts — correct pattern
if (typeof window !== 'undefined') {
  app.component('ConvergenceChart', defineAsyncComponent(() =>
    import('./components/ConvergenceChart.vue')
  ))
}
```

Always wrap chart components in `<ClientOnly>` in markdown.

### Benchmark Data Flow
```
benchmarks/run_benchmark_suite.py
  → benchmarks/output/results.json      (CI artifact)
  → docs/public/benchmarks/*.json       (served to browser)
      ↓
  benchmarkTransforms.ts                (buildConvergenceSeries / buildECDFSeries / buildViolinSeries)
      ↓
  ConvergenceChart / ECDFChart / ViolinPlot (ECharts, Catppuccin Mocha theme)
```

### API Documentation Flow
```
src/opt/ (Python source)
  → griffe dump → docs/api/*.json
      ↓
  docs/.vitepress/loaders/api.data.ts   (VitePress data loader)
      ↓
  APIDoc.vue (rendered on algorithm pages)
```

## GitHub / git

- **Main branch**: `main`
- **Active epic**: Issue #52 — VitePress Documentation Site with Scientific Visualization Suite

### gh CLI
Must unset the environment token before use:
```bash
unset GITHUB_TOKEN && gh issue view 52 --repo Anselmoo/useful-optimizer
```
Alternatively use the `mcp__github__*` MCP tools which bypass this conflict.

## Open Work (Epic #52)

| Issue | Description | Status |
|-------|-------------|--------|
| #134 | Wire ECharts to real benchmark data | ✅ Done |
| #92 | package-lock.json for npm CI | ✅ Done (docs/package-lock.json present) |
| #86 | CI/CD & GitHub Pages deployment | ✅ In progress — PR #145 |
| #55 | Zensical alternative (alternative doc stack) | Parked |
