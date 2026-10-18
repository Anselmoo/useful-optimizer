# Benchmark Methodology

All single-objective optimizers of the library are benchmarked with one protocol, modelled on
[COCO](https://github.com/numbbo/coco) and IOHprofiler: every run gets the **same budget of
function evaluations**, results are measured against evaluations, and every statement of
"better" is backed by a statistical test against a baseline. The code is in
[`benchmarks/`](https://github.com/Anselmoo/useful-optimizer/tree/main/benchmarks) and the data it
produces is described by a [JSON schema](https://github.com/Anselmoo/useful-optimizer/blob/main/docs/schemas/benchmark-data-schema.json).

## Protocol

| Item | Value |
|------|-------|
| Optimizers | every single-objective optimizer of `opt` (see [Scope](#scope)) |
| Functions | Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, each with a shifted optimum |
| Dimensions | 2, 5, 10, 20 |
| Independent runs | 15 per optimizer, function and dimension |
| Budget | $2000 \cdot n$ function evaluations per run ($n$ = dimension) |
| Targets | precision $f - f^* \le 10^{1}, 10^{0}, \dots, 10^{-8}$ (one per decade) |
| Seeds | run $r$ on function number $k$ uses seed $r + 100k$; the same seeds for every optimizer |
| Baseline | `RandomSearch`: uniform sampling in the search domain |

### Equal evaluation budget

Optimizers differ enormously in the cost of one iteration (from a fraction of an evaluation to
thousands), so a common `max_iter` is not a common budget. The objective is therefore wrapped in a
counter that stops the run when the budget is used up (with an exception that optimizers cannot
catch by accident), and `max_iter` is chosen from two short probe runs so that the optimizer's own
schedule (cooling, inertia, step sizes) just fits the budget. Everything is reported against the
number of evaluations the wrapper counted, not the optimizer's own bookkeeping.

Hyperparameters other than `max_iter`, `population_size` (30 where an optimizer has one) and the
seed are the library defaults. No optimizer is tuned.

### Time caps

A run that exceeds 20 s of wall-clock time is cut off and counted with the best value found until
then (`truncated` in the statistics table); a cell stops starting new runs after 150 s but always
finishes at least five. This only affects a few expensive optimizers (for example Bayesian
optimization) and is visible in the data, never hidden.

### Scope

The suite covers optimizers that minimise one scalar function. The multi-objective optimizers
(`MOEAD`, `NSGAII`, `SPEA2`) return a Pareto front, which needs different problems and indicators;
their pages say so. The constrained optimizers are benchmarked on the unconstrained functions.

### Search domains and shifted optima

| Function | Domain | $f^*$ |
|----------|--------|-------|
| Sphere | $[-5.12, 5.12]^n$ | 0 |
| Rosenbrock | $[-5, 10]^n$ | 0 |
| Rastrigin | $[-5.12, 5.12]^n$ | 0 |
| Ackley | $[-32.768, 32.768]^n$ | 0 |
| Griewank | $[-600, 600]^n$ | 0 |

See [Benchmark Functions](./functions) for the definitions.

The optimizers do not minimise $f(\mathbf{x})$ but $f(\mathbf{x} - \mathbf{s})$, where the shift
$\mathbf{s}$ places the optimum uniformly inside 80% of the half-width of the domain around its
centre, as COCO does for its instances. It is fixed per function and dimension, the same for every
optimizer and run, and published in `index.json`. Without it, an optimizer that averages, scales or
shrinks positions towards the origin or the centre of the box finds the optimum of Sphere, Rastrigin
or Ackley by construction: in a test, `ArithmeticOptimizationAlgorithm` and
`SeagullOptimizationAlgorithm` reached exactly 0 on the unshifted 10D sphere but 5.9 and 11.4 once the
optimum was moved.

## Metrics

All metrics are computed from the unrounded raw runs, then rounded to four significant digits for
publication.

**Precision** is $f - f^*$ of the best point found. Reported per run as the final value, and as the
median and interquartile range over runs at 60 log-spaced evaluation counts (the convergence
curves).

**Success rate** at a target is the fraction of runs that reached it at any time within the budget.

**ERT** (expected running time, COCO) for one target:

$$
\mathrm{ERT} = \frac{\sum_{i \in \text{success}} T_i + \sum_{i \in \text{failure}} B_i}{N_{\text{success}}}
$$

where $T_i$ is the number of evaluations until run $i$ first reached the target and $B_i$ the
evaluations a failed run used. It is the expected number of evaluations a restarted optimizer needs
per success, and is undefined if no run succeeded. **SP1** is the mean number of evaluations of the
successful runs divided by the success rate.

**Runtime ECDF** is the fraction of (run, target) pairs reached within a given number of
evaluations per dimension, over all ten targets (40 log-spaced budgets).

**Cost** is the median wall-clock time per run and per evaluation. It includes the optimizer's own
overhead, which is the point: a cheap-looking algorithm may be expensive per evaluation.

**Dispersion of evaluated points** is the standard deviation of the 30 most recently evaluated points
(per coordinate, averaged over coordinates) divided by the width of the search domain. It is a proxy
that works for every optimizer because it only uses the evaluated points. It is **not** the
diversity of a population, which most implementations do not expose.

## Statistical analysis

### Test against the baseline

For every optimizer and every function and dimension, the final precisions of its runs are compared with
those of `RandomSearch` by a one-sided **Mann-Whitney U test** (alternative: the optimizer has the
smaller error). Runs are independent, not paired, so the rank-sum test applies, not the signed-rank test.
Errors below $10^{-8}$ are floored so that solved runs tie.

The $p$-values of all optimizers of one function and dimension are adjusted with the
**Holm–Bonferroni** step-down method (family-wise error rate), and the tables mark the adjusted values
with `*` ($< 0.05$), `**` ($< 0.01$) and `***` ($< 0.001$).

With 15 runs the test cannot detect small differences. The **effect size** is reported next to the
$p$-value as **Cliff's delta**,

$$
\delta = \frac{\#(x_i < y_j) - \#(x_i > y_j)}{n_x n_y} \in [-1, 1],
$$

where $x$ are the optimizer's errors and $y$ the baseline's: $+1$ means every run beat every baseline
run.

### Average ranks and the Friedman test

In every function–dimension block the optimizers are ranked by the median of their (floored) final
precision, rank 1 being best and ties sharing the mean rank. The **average rank** over the blocks is
the headline number of the [results page](./results). The **Friedman test** over the blocks in which
every optimizer has a result checks whether the differences in average rank can be due to chance:

$$
\chi_F^2 = \frac{12N}{k(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4} \right]
$$

where $N$ is the number of blocks, $k$ the number of optimizers and $R_j$ the average rank of
optimizer $j$. A significant Friedman test says that the optimizers differ somewhere; it does not say
which pairs differ. For that use the per-block tests against the baseline.

## Limitations

- **Budget**: $2000 \cdot n$ evaluations favours optimizers that converge quickly; some
  population-based methods are designed for much longer runs. Rankings can change with the budget.
- **Default parameters**: the comparison measures the library defaults, not the best each method can do.
- **Five functions**: the results say little about problems unlike these (noise, constraints,
  expensive evaluations, structure).
- **15 runs**: enough to see large effects; small differences stay inconclusive, which is why the effect
  size is reported.
- **Shared seeds** make comparisons reproducible, but runs of different optimizers are still
  independent samples.
- **Hardware**: wall-clock costs depend on the machine the benchmark ran on (listed in `index.json`).

## Reproducibility

```bash
uv sync --all-extras

# a quick local check: 4 optimizers, one function, 5 runs
uv run python benchmarks/run_benchmark_suite.py --tier showcase \
    --functions sphere --dimensions 2,5 --n-runs 5 --output-dir benchmarks/output

# the whole suite, one category and function at a time, resumable
uv run python benchmarks/run_benchmark_suite.py --tier comprehensive \
    --category swarm_intelligence --functions rastrigin --workers 4 --resume

# merge the raw cells into the published data
uv run python benchmarks/aggregate_results.py --input benchmarks/output \
    --output-dir benchmarks/output/processed
```

Every finished cell is written to its own file (`raw/<function>/<dimension>/<optimizer>.json`), so an
interrupted run loses nothing. The [benchmark pipeline](https://github.com/Anselmoo/useful-optimizer/actions/workflows/benchmark-pipeline.yml)
runs the same commands, one job per category and function.

## References

1. Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. "COCO: A platform for comparing continuous optimizers in a black-box setting." *Optimization Methods and Software* 36(1), 2021. arXiv:1603.08785.
2. Doerr, C., Wang, H., Ye, F., van Rijn, S., Bäck, T. "IOHprofiler: A benchmarking and profiling tool for iterative optimization heuristics." arXiv:1810.05281 (2018).
3. Demšar, J. "Statistical comparisons of classifiers over multiple data sets." *Journal of Machine Learning Research* 7 (2006): 1–30.
4. Holm, S. "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics* 6(2) (1979): 65–70.
