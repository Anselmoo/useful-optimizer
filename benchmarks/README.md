# Benchmark Suite

Benchmarks every single-objective optimizer of `useful-optimizer` under one protocol: the same
budget of function evaluations for every run, fixed seeds, and statistics against a random search
baseline. The protocol, formulas and limitations are described in
[`docs/benchmarks/methodology.md`](../docs/benchmarks/methodology.md); this page is about the code.

## Files

| File | Purpose |
|------|---------|
| `protocol.py` | Evaluation-budget protocol: `BudgetedObjective`, optimizer discovery and construction, one run, one cell |
| `run_benchmark_suite.py` | Runner: tiers, shards, worker processes, resumable raw output |
| `metrics.py` | ERT/SP1, ECDF, quantile curves, Holm, Mann-Whitney U, Cliff's delta, ranks, Friedman test |
| `models.py` | Pydantic models of the published data (source of truth for the schema) |
| `aggregate_results.py` | Raw cells → published data (`index.json`, `cells/`, `ranks.json`, `trajectory-*.json`) |
| `export_schema.py` | Writes `docs/schemas/benchmark-data-schema.json` and, with `--types`, the TypeScript types |
| `test/` | Tests of the protocol, runner, metrics and the aggregation |
| `output/` | Local results (gitignored) |

## Quick start

```bash
uv sync --all-extras

# a quick local check
uv run python benchmarks/run_benchmark_suite.py --tier showcase \
    --functions sphere --dimensions 2,5 --n-runs 5 --output-dir benchmarks/output

# aggregate and validate
uv run python benchmarks/aggregate_results.py --input benchmarks/output \
    --output-dir benchmarks/output/processed
```

## Tiers

| Tier | Optimizers | Use |
|------|-----------|-----|
| `showcase` | 4 (`ParticleSwarm`, `DifferentialEvolution`, `AdamW`, `HarmonySearch`) | local checks, weekly pipeline smoke test |
| `standard` | 13 across all major families | a representative sample |
| `comprehensive` | every single-objective optimizer found in `opt` | the published data |

`RandomSearch`, the baseline of all statistical tests, is part of every tier. The unit of work is a
**cell**: one optimizer on one function and dimension with `--n-runs` runs. Each finished cell is
written to `raw/<function>/<dimension>/<optimizer>.json`, so an interrupted run loses nothing.

## Runner options

```
--tier {showcase,standard,comprehensive}
--category CAT[,CAT]       only these optimizer categories
--optimizers NAME[,NAME]   only these optimizers
--functions NAME[,NAME]    default: all six
--dimensions D[,D]         default: 2,5,10,20
--n-runs N                 default: 15
--budget-per-dim N         default: 2000
--time-cap SECONDS         per run (default 20)
--cell-time SECONDS        per cell (default 150, at least 5 runs always complete)
--shard i/n                run slice i of n of the selected cells
--workers N                parallel processes
--resume                   skip finished cells
--list-shards              print the CI job matrix (category x function) and exit
--dry-run                  count the cells and exit
```

## CI

`.github/workflows/benchmark-pipeline.yml` runs one job per category and function (`--list-shards`),
uploads the raw cells of each job, and aggregates them in a final job. Scheduled runs (a monthly
comprehensive run and a weekly showcase smoke test) only produce an artifact. A manual run with
`publish` set commits the aggregated data to `docs/public/benchmarks/` and redeploys the docs.

## Tests

```bash
uv run pytest benchmarks/test -o addopts=""
```

They check that the budget cut-off is exact and cannot be swallowed, that the wrapper's evaluation
count matches the optimizer's own for every optimizer, that runs differ across seeds and repeat for
one seed, the metrics against hand-computed COCO examples, that the shards partition the cells, and
that the schema file matches the models.

## Changing the data format

Edit `models.py`, then regenerate the schema and the TypeScript types and commit all three:

```bash
uv run python benchmarks/export_schema.py --types
```

`test_aggregate.py` fails when the committed schema drifts from the models.
