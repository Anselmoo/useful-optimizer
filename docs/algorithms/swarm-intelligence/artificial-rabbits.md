# Artificial Rabbits Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Artificial Rabbits Optimization (ARO) Algorithm.

## Algorithm Overview

This module implements the Artificial Rabbits Optimization algorithm,
a bio-inspired metaheuristic based on the survival strategies of rabbits.

Rabbits exhibit two main survival behaviors: detour foraging (moving
irregularly to avoid predators) and random hiding (seeking shelter).

## Reference

> Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082. DOI: 10.1016/j.engappai.2022.105082

[📄 View Paper (DOI: 10.1016/j.engappai.2022.105082)](https://doi.org/10.1016/j.engappai.2022.105082)

## Usage

```python
from opt.swarm_intelligence.artificial_rabbits import ArtificialRabbitsOptimizer
from opt.benchmark.functions import sphere

optimizer = ArtificialRabbitsOptimizer(
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
| `population_size` | `int` | `30` | Population size. |
| `max_iter` | `int` | `100` | Maximum iterations. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`artificial_rabbits.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/artificial_rabbits.py)
:::
