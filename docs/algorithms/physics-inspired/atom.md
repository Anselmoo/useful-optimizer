# Atom Search Optimizer

<span class="badge badge-physics">Physics-Inspired</span>

Atom Search Optimization (ASO).

## Algorithm Overview

This module implements Atom Search Optimization, a physics-inspired
metaheuristic algorithm based on molecular dynamics simulation.

## Reference

> Zhao, W., Wang, L., & Zhang, Z. (2019). Atom search optimization and its application to solve a hydrogeologic parameter estimation problem. Knowledge-Based Systems, 163, 283-304.

## Usage

```python
from opt.physics_inspired.atom_search import AtomSearchOptimizer
from opt.benchmark.functions import sphere

optimizer = AtomSearchOptimizer(
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
| `population_size` | `int` | `50` | Population size (number of atoms). |
| `max_iter` | `int` | `500` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Physics-Inspired Algorithms](/algorithms/physics-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`atom_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/physics_inspired/atom_search.py)
:::
