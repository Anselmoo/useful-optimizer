# Manta Ray Foraging Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Manta Ray Foraging Optimization (MRFO) implementation.

## Algorithm Overview

This module implements the Manta Ray Foraging Optimization algorithm, a
nature-inspired metaheuristic based on the foraging behaviors of manta rays.

## Reference

> Zhao, W., Zhang, Z., & Wang, L. (2020). Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, 103300.

## Usage

```python
from opt.swarm_intelligence.manta_ray import MantaRayForagingOptimization
from opt.benchmark.functions import sphere

optimizer = MantaRayForagingOptimization(
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
| `func` | `Callable` | Required | Objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `max_iter` | `int` | Required | Maximum number of iterations |
| `population_size` | `int` | `30` | Number of individuals in population |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`manta_ray.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/manta_ray.py)
:::
