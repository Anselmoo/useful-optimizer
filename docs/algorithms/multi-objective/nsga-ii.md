# NSGA-II

<span class="badge badge-multi">Multi-Objective</span>

NSGA-II: Non-dominated Sorting Genetic Algorithm II.

## Algorithm Overview

This module implements the NSGA-II algorithm, one of the most popular and
highly-cited multi-objective evolutionary optimization algorithms.

NSGA-II uses fast non-dominated sorting and crowding distance assignment
to maintain a well-spread Pareto-optimal front while efficiently converging
to the true Pareto front.

## Reference

> Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197. DOI: 10.1109/4235.996017

[📄 View Paper (DOI: 10.1109/4235.996017)](https://doi.org/10.1109/4235.996017)

## Usage

```python
from opt.multi_objective.nsga_ii import NSGAII
from opt.benchmark.functions import sphere

optimizer = NSGAII(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
    population_size=50,
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objectives` | `Sequence[Callable]` | Required | List of objective functions
        to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `max_iter` | `int` | `200` | Maximum number of generations. |
| `seed` | `int  \|  None` | `None` | **REQUIRED for BBOB compliance. |
| `population_size` | `int` | `100` | Number of individuals in population. |
| `crossover_prob` | `float` | `_CROSSOVER_PROBABILITY` | SBX crossover probability. |
| `mutation_prob` | `float  \|  None` | `None` | Polynomial mutation probability per
        dimension. |
| `tournament_size` | `int` | `_TOURNAMENT_SIZE` | Binary tournament selection size. |
| `eta_c` | `float` | `_SBX_DISTRIBUTION_INDEX` | SBX distribution index. |
| `eta_m` | `float` | `_POLYNOMIAL_MUTATION_INDEX` | Polynomial mutation distribution index. |

## See Also

- [Multi-Objective Algorithms](/algorithms/multi-objective/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`nsga_ii.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/multi_objective/nsga_ii.py)
:::
