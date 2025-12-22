# Tabu Search

<span class="badge badge-classical">Classical</span>

Tabu Search.

## Algorithm Overview

This module implements the Tabu Search optimization algorithm.

The Tabu Search algorithm is a metaheuristic optimization algorithm that is used to
solve combinatorial optimization problems. It is inspired by the concept of memory in
human search behavior. The algorithm maintains a tabu list that keeps track of recently
visited solutions and prevents the search from revisiting them in the near future. This
helps the algorithm to explore different regions of the search space and avoid getting
stuck in local optima.

This module provides the `TabuSearch` class, which is an implementation of the
Tabu Search algorithm. It can be used to minimize a given objective function
over a continuous search space.

## Usage

```python
from opt.classical.tabu_search import TabuSearch
from opt.benchmark.functions import sphere

optimizer = TabuSearch(
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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `population_size` | `int` | `100` | The size of the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `tabu_list_size` | `int` | `50` | The size of the tabu list. |
| `neighborhood_size` | `int` | `10` | The size of the neighborhood. |
| `seed` | `int  \|  None` | `None` | The seed for the random number generator. |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`tabu_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/tabu_search.py)
:::
