# Benchmark Tier System

A tier selects which optimizers a benchmark run covers. All tiers use the same
[protocol](../docs/benchmarks/methodology.md) (equal evaluation budget, 15 runs, 5 functions (shifted optima),
dimensions 2, 5, 10 and 20) and always include the baseline `RandomSearch`, because the statistical
tests compare against it.

| Tier | Optimizers | Use |
|------|-----------|-----|
| **showcase** | 4 | quick local checks, weekly smoke test of the pipeline |
| **standard** | 13 | a representative sample across the families |
| **comprehensive** | every single-objective optimizer (115 with the baseline) | the data published on the docs site |

```bash
uv run python benchmarks/run_benchmark_suite.py --tier showcase
uv run python benchmarks/run_benchmark_suite.py --tier standard
uv run python benchmarks/run_benchmark_suite.py --tier comprehensive --workers 4 --resume
```

## Why the comprehensive tier is feasible

Earlier versions ran a fixed number of iterations, which made the cost of an optimizer depend on how
many evaluations its iteration happens to use (from a fraction of an evaluation to thousands). The
protocol now gives every run the same budget of $2000 \cdot n$ evaluations and cuts the run off at it,
so the cost per cell is bounded. Costly optimizers are additionally limited by a wall-clock cap per run and
per cell and are flagged in the data when it applies.

The full suite is 115 optimizers x 5 functions x 4 dimensions x 15 runs. The pipeline splits it into
one job per category and function (45 jobs), each of which writes every finished cell to its own file.
A job that is cut off loses nothing and `--resume` continues it.

## Tiers

**showcase**: `ParticleSwarm`, `DifferentialEvolution`, `AdamW`, `HarmonySearch`.

**standard**:

- Swarm intelligence: `ParticleSwarm`, `AntColony`, `FireflyAlgorithm`, `BatAlgorithm`, `GreyWolfOptimizer`
- Evolutionary: `GeneticAlgorithm`, `DifferentialEvolution`
- Metaheuristic: `HarmonySearch`, `SimulatedAnnealing`
- Classical: `HillClimbing`, `NelderMead`
- Gradient-based: `AdamW`, `SGDMomentum`

**comprehensive**: every `AbstractOptimizer` subclass found by `benchmarks.protocol.discover_optimizers()`.
The multi-objective optimizers (`MOEAD`, `NSGAII`, `SPEA2`) are not part of it because they return a Pareto
front instead of one value.

See `benchmarks/README.md` for the runner options and the CI pipeline.
