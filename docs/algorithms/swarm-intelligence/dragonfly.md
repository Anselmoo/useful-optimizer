# Dragonfly Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Dragonfly Algorithm (DA).

## Algorithm Overview

This module implements the Dragonfly Algorithm, a swarm intelligence optimization
algorithm based on the static and dynamic swarming behaviors of dragonflies.

Dragonflies form sub-swarms for hunting (static swarm) and migrate in one direction
(dynamic swarm). These behaviors map to exploration and exploitation in optimization.

## Reference

> Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems. Neural Computing and Applications, 27(4), 1053-1073. DOI: 10.1007/s00521-015-1920-1

[📄 View Paper (DOI: 10.1007/s00521-015-1920-1)](https://doi.org/10.1007/s00521-015-1920-1)

## Usage

```python
from opt.swarm_intelligence.dragonfly_algorithm import DragonflyOptimizer
from opt.benchmark.functions import sphere

optimizer = DragonflyOptimizer(
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
| `population_size` | `int` | `100` | Number of dragonflies in swarm. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`dragonfly_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/dragonfly_algorithm.py)
:::
