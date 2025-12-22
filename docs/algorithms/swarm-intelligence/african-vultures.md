# African Vultures Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

African Vultures Optimization Algorithm (AVOA).

## Algorithm Overview

This module implements the African Vultures Optimization Algorithm,
a nature-inspired metaheuristic based on the foraging and navigation
behaviors of African vultures.

## Reference

> Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). African vultures optimization algorithm: A new nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering, 158, 107408.

## Usage

```python
from opt.swarm_intelligence.african_vultures_optimizer import AfricanVulturesOptimizer
from opt.benchmark.functions import sphere

optimizer = AfricanVulturesOptimizer(
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
| `population_size` | `int` | `50` | Number of individuals in population |
| `max_iter` | `int` | `500` | Maximum number of iterations |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`african_vultures_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/african_vultures_optimizer.py)
:::
