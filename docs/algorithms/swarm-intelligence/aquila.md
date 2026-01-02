# Aquila Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Aquila Optimizer (AO).

## Algorithm Overview

This module implements the Aquila Optimizer, a nature-inspired
metaheuristic algorithm based on the hunting behavior of Aquila
(eagle) in nature.

## Reference

> Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A. A., Al-qaness, M. A., & Gandomi, A. H. (2021). Aquila optimizer: A novel meta-heuristic optimization algorithm. Computers & Industrial Engineering, 157, 107250.

## Usage

```python
from opt.swarm_intelligence.aquila_optimizer import AquilaOptimizer
from opt.benchmark.functions import sphere

optimizer = AquilaOptimizer(
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
View the implementation: [`aquila_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/aquila_optimizer.py)
:::
