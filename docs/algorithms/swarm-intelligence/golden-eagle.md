# Golden Eagle Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Golden Eagle Optimizer (GEO) implementation.

## Algorithm Overview

This module implements the Golden Eagle Optimizer, a nature-inspired
metaheuristic based on the intelligent hunting behavior of golden eagles.

## Reference

> Mohammadi-Balani, A., Nayeri, M. D., Azar, A., & Taghizadeh-Yazdi, M. (2021). Golden eagle optimizer: A nature-inspired metaheuristic algorithm. Computers & Industrial Engineering, 152, 107050.

## Usage

```python
from opt.swarm_intelligence.golden_eagle import GoldenEagleOptimizer
from opt.benchmark.functions import sphere

optimizer = GoldenEagleOptimizer(
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
View the implementation: [`golden_eagle.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/golden_eagle.py)
:::
