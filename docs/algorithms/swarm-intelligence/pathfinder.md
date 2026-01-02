# Pathfinder Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Pathfinder Algorithm (PFA) implementation.

## Algorithm Overview

This module implements the Pathfinder Algorithm, a swarm-based
metaheuristic optimization algorithm inspired by the collective
movement of animal groups searching for food.

## Reference

> Yapici, H., & Cetinkaya, N. (2019). A new meta-heuristic optimizer: Pathfinder algorithm. Applied Soft Computing, 78, 545-568.

## Usage

```python
from opt.swarm_intelligence.pathfinder import PathfinderAlgorithm
from opt.benchmark.functions import sphere

optimizer = PathfinderAlgorithm(
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
| `max_iter` | `int` | Required | Maximum iterations. |
| `population_size` | `int` | `30` | Population size. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`pathfinder.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/pathfinder.py)
:::
