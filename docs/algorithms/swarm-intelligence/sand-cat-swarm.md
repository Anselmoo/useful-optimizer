# Sand Cat Swarm Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Sand Cat Swarm Optimization Algorithm.

## Algorithm Overview

Implementation based on:
Seyyedabbasi, A. & Kiani, F. (2023).
Sand Cat swarm optimization: A nature-inspired algorithm to solve
global optimization problems.
Engineering with Computers, 39(4), 2627-2651.

The algorithm mimics the hunting behavior of sand cats, small wild cats
that are efficient hunters in desert environments.

## Usage

```python
from opt.swarm_intelligence.sand_cat import SandCatSwarmOptimizer
from opt.benchmark.functions import sphere

optimizer = SandCatSwarmOptimizer(
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
View the implementation: [`sand_cat.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/sand_cat.py)
:::
