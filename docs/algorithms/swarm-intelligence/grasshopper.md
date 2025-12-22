# Grasshopper Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Grasshopper Optimization Algorithm (GOA).

## Algorithm Overview

This module implements the Grasshopper Optimization Algorithm, a nature-inspired
metaheuristic based on the swarming behavior of grasshoppers in nature.

Grasshoppers naturally form swarms and move toward food sources while avoiding
collisions with each other. The algorithm mimics this behavior with social forces
(attraction/repulsion) and movement toward the best solution.

## Reference

> Saremi, S., Mirjalili, S., & Lewis, A. (2017). Grasshopper Optimisation Algorithm: Theory and application. Advances in Engineering Software, 105, 30-47. DOI: 10.1016/j.advengsoft.2017.01.004

[📄 View Paper (DOI: 10.1016/j.advengsoft.2017.01.004)](https://doi.org/10.1016/j.advengsoft.2017.01.004)

## Usage

```python
from opt.swarm_intelligence.grasshopper_optimization import GrasshopperOptimizer
from opt.benchmark.functions import sphere

optimizer = GrasshopperOptimizer(
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
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |
| `population_size` | `int` | `100` | Number of individuals in population |
| `c_max` | `float` | `_C_MAX` | Algorithm-specific parameter |
| `c_min` | `float` | `_C_MIN` | Algorithm-specific parameter |
| `f` | `float` | `_ATTRACTION_INTENSITY` | Algorithm-specific parameter |
| `l` | `float` | `_ATTRACTIVE_LENGTH_SCALE` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`grasshopper_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/grasshopper_optimization.py)
:::
