# Variable Depth Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Variable Depth Search (VDS) Algorithm.

## Algorithm Overview

This module implements the Variable Depth Search (VDS) optimization algorithm. VDS is a
local search method used for mathematical optimization. It explores the search space by
variable-depth first search and backtracking.

The main idea behind VDS is to perform a search in a variable depth to find the optimal
solution for a given function. The depth of the search is defined by the `depth`
parameter. The larger the depth, the more potential solutions the algorithm will
consider at each step, but the more computational resources it will require.

VDS is particularly useful for problems where the search space is large and complex, and
where traditional optimization methods may not be applicable.

## Usage

```python
from opt.metaheuristic.variable_depth_search import VariableDepthSearch
from opt.benchmark.functions import sphere

optimizer = VariableDepthSearch(
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
| `max_depth` | `int` | `20` | The maximum depth of the search. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`variable_depth_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/variable_depth_search.py)
:::
