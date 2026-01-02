# Squirrel Search Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Squirrel Search Algorithm.

## Algorithm Overview

!!! warning

    This module is still under development and is not yet ready for use.

## Usage

```python
from opt.swarm_intelligence.squirrel_search import SquirrelSearchAlgorithm
from opt.benchmark.functions import sphere

optimizer = SquirrelSearchAlgorithm(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Population size. |
| `track_history` | `bool` | `False` | Enable convergence history tracking for BBOB
        post-processing. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`squirrel_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/squirrel_search.py)
:::
