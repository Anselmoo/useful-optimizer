# Benchmarks Overview

Every single-objective optimizer of the library is benchmarked with one protocol, following the
standards of [COCO](https://github.com/numbbo/coco) and
[IOHprofiler](https://iohprofiler.github.io/): the same budget of function evaluations for every
run, independent runs with fixed seeds, runtime distributions, and statistical tests against a
random search baseline.

## Benchmark suite

- **Optimizers**: all single-objective optimizers of `opt`, plus [`RandomSearch`](/algorithms/classical/random-search) as the baseline
- **5 functions**: Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, each with the optimum moved away from the centre of the domain
- **4 dimensions**: 2, 5, 10, 20
- **15 independent runs** per optimizer, function and dimension
- **Budget**: 2000 × dimension function evaluations per run

## Where to look

### [Results](./results)

The average rank of every optimizer over all functions and dimensions with the Friedman test, and the
interactive panel described below.

### Every algorithm page

Each [algorithm page](/algorithms/) ends with the benchmark panel for that optimizer. Choose a
function and a dimension, add optimizers to compare with, and switch between:

| Tab | Shows |
|-----|-------|
| Convergence | median precision against evaluations per dimension, with the interquartile band |
| Distribution | kernel density, boxplot and individual runs of the final precision |
| ECDF | share of (run, target) pairs solved within a budget (COCO runtime ECDF) |
| Scaling | median precision, expected running time or cost against the dimension |
| Statistics | median, IQR, success rate, ERT, SP1, cost, Holm-adjusted p-value and Cliff's delta against the baseline; CSV export |
| Dispersion | spread of the recently evaluated points |
| Landscape (2D) | the best-so-far path of a run on the 3D surface of the function |

### [Methodology](./methodology)

The protocol, the formulas of every metric, the statistical tests and the limitations.

### [Benchmark Functions](./functions)

Definitions, domains and optima of the test functions.

### [Visualization Architecture](./visualization)

How the charts are built, for contributors.

## Running benchmarks

```bash
# a quick local check
uv run python benchmarks/run_benchmark_suite.py --tier showcase \
    --functions sphere --dimensions 2,5 --n-runs 5 --output-dir benchmarks/output

# aggregate the raw cells into the published data format
uv run python benchmarks/aggregate_results.py --input benchmarks/output \
    --output-dir benchmarks/output/processed
```

The full suite runs in the [benchmark pipeline](https://github.com/Anselmoo/useful-optimizer/actions/workflows/benchmark-pipeline.yml),
one job per category and function; see [`benchmarks/README.md`](https://github.com/Anselmoo/useful-optimizer/blob/main/benchmarks/README.md).

## Data format

The browser reads `/benchmarks/index.json` (protocol, optimizers, available cells), one file per
function and dimension (`/benchmarks/cells/<function>-<dimension>.json`), `/benchmarks/ranks.json`
and `/benchmarks/trajectory-<function>-2.json`. The format is defined by the Pydantic models in
`benchmarks/models.py`; the
[JSON schema](https://github.com/Anselmoo/useful-optimizer/blob/main/docs/schemas/benchmark-data-schema.json)
and the TypeScript types are generated from them.

## References

- [COCO Platform](https://github.com/numbbo/coco) - Comparing Continuous Optimizers
- [IOHprofiler](https://iohprofiler.github.io/) - Iterative Optimization Heuristics Profiler
- [IOHanalyzer](https://iohanalyzer.liacs.nl/) - Interactive performance analysis
