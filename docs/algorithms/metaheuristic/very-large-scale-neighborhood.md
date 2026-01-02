# Very Large Scale Neighborhood

<span class="badge badge-metaheuristic">Metaheuristic</span>

Very Large Scale Neighborhood Search (VLSN) Algorithm.

## Algorithm Overview

This module implements the Very Large Scale Neighborhood Search (VLSN) optimization
algorithm. VLSN is a local search method used for mathematical optimization.
It explores very large neighborhoods with an efficient algorithm.

The main idea behind VLSN is to perform a search in a large-scale neighborhood to find
the optimal solution for a given function. The size of the neighborhood is defined by
the `neighborhood_size` parameter.
The larger the neighborhood size, the more potential solutions the algorithm will
consider at each step, but the more computational resources it will require.

VLSN is particularly useful for problems where the search space is large and complex,
and where traditional optimization methods may not be applicable.

## Usage

```python
from opt.metaheuristic.very_large_scale_neighborhood_search import VeryLargeScaleNeighborhood
from opt.benchmark.functions import sphere

optimizer = VeryLargeScaleNeighborhood(
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
| `population_size` | `int` | `100` | Number of individuals in population. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `neighborhood_size` | `int` | `10` | Number of neighbors explored around each
        individual. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`very_large_scale_neighborhood_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/very_large_scale_neighborhood_search.py)
:::
