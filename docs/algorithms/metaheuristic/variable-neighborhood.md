# Variable Neighborhood Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Variable Neighborhood Search optimizer.

## Algorithm Overview

This module implements the Variable Neighborhood Search (VNS) optimizer. VNS is a
metaheuristic optimization algorithm that explores different neighborhoods of a
solution to find the optimal solution for a given objective function within a specified
search space.

The `VariableNeighborhoodSearch` class is the main class that implements the VNS
algorithm. It takes an objective function, lower and upper bounds of the search space,
dimensionality of the search space, and other optional parameters to control the
optimization process.

## Usage

```python
from opt.metaheuristic.variable_neighbourhood_search import VariableNeighborhoodSearch
from opt.benchmark.functions import sphere

optimizer = VariableNeighborhoodSearch(
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
| `func` | `Callable` | Required | Objective function to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `population_size` | `int` | `100` | Number of candidate solutions. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `neighborhood_size` | `int` | `10` | Maximum neighborhood depth (k_max). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`variable_neighbourhood_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/variable_neighbourhood_search.py)
:::
