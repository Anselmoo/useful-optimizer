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
| `func` | `Callable` | Required | The objective function to be optimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `population_size` | `int` | `100` | The size of the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `neighborhood_size` | `int` | `10` | The size of the neighborhood. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`variable_neighbourhood_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/variable_neighbourhood_search.py)
:::
