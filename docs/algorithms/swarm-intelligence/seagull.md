# Seagull Optimization Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Seagull Optimization Algorithm (SOA) implementation.

## Algorithm Overview

This module implements the Seagull Optimization Algorithm, a bio-inspired
metaheuristic based on the migration and attack behavior of seagulls.

## Reference

> Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems. Knowledge-Based Systems, 165, 169-196.

## Usage

```python
from opt.swarm_intelligence.seagull_optimization import SeagullOptimizationAlgorithm
from opt.benchmark.functions import sphere

optimizer = SeagullOptimizationAlgorithm(
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
View the implementation: [`seagull_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/seagull_optimization.py)
:::
