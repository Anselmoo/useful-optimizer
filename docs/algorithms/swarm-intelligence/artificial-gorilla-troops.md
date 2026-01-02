# Artificial Gorilla Troops Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Artificial Gorilla Troops Optimizer (GTO).

## Algorithm Overview

This module implements the Artificial Gorilla Troops Optimizer,
a metaheuristic algorithm inspired by the social intelligence
of gorilla troops in nature.

## Reference

> Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). Artificial gorilla troops optimizer: A new nature-inspired metaheuristic algorithm for global optimization problems. International Journal of Intelligent Systems, 36(10), 5887-5958.

## Usage

```python
from opt.swarm_intelligence.artificial_gorilla_troops import ArtificialGorillaTroopsOptimizer
from opt.benchmark.functions import sphere

optimizer = ArtificialGorillaTroopsOptimizer(
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
| `population_size` | `int` | `50` | Population size. |
| `max_iter` | `int` | `500` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`artificial_gorilla_troops.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/artificial_gorilla_troops.py)
:::
