# Moth Flame Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Moth-Flame Optimization (MFO) Algorithm.

## Algorithm Overview

This module implements the Moth-Flame Optimization algorithm, a nature-inspired
metaheuristic based on the navigation behavior of moths in nature.

Moths use a mechanism called transverse orientation for navigation. They maintain
a fixed angle with respect to the moon (a distant light source). However, when moths
encounter artificial lights, this mechanism leads to spiral flight paths around flames.

## Reference

> Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm. Knowledge-Based Systems, 89, 228-249. DOI: 10.1016/j.knosys.2015.07.006

[📄 View Paper (DOI: 10.1016/j.knosys.2015.07.006)](https://doi.org/10.1016/j.knosys.2015.07.006)

## Usage

```python
from opt.swarm_intelligence.moth_flame_optimization import MothFlameOptimizer
from opt.benchmark.functions import sphere

optimizer = MothFlameOptimizer(
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
| `population_size` | `int` | `100` | Number of moths/flames. |
| `b` | `float` | `1.0` | Spiral shape constant. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`moth_flame_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/moth_flame_optimization.py)
:::
