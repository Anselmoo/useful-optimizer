# MOEA/D

<span class="badge badge-multi">Multi-Objective</span>

MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition).

## Algorithm Overview

This module implements MOEA/D, a highly influential multi-objective
optimization algorithm that decomposes a multi-objective problem into
scalar subproblems.

## Reference

> Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 712-731.

## Usage

```python
from opt.multi_objective.moead import MOEAD
from opt.benchmark.functions import sphere

optimizer = MOEAD(
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
| `objectives` | `list[Callable]` | Required | List of objective functions
        to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `population_size` | `int` | `100` | Number of subproblems (weight vectors). |
| `max_iter` | `int` | `300` | Maximum number of generations. |
| `n_neighbors` | `int` | `20` | Neighborhood size for each subproblem. |

## See Also

- [Multi-Objective Algorithms](/algorithms/multi-objective/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`moead.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/multi_objective/moead.py)
:::
