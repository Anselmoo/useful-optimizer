# Pelican Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Pelican Optimization Algorithm (POA).

## Algorithm Overview

This module implements the Pelican Optimization Algorithm, a bio-inspired
metaheuristic based on the hunting behavior of pelicans.

Pelicans are known for their cooperative hunting strategies, including
group fishing and synchronized diving to catch prey.

## Reference

> Trojovský, P., & Dehghani, M. (2022). Pelican Optimization Algorithm: A Novel Nature-Inspired Algorithm for Engineering Applications. Sensors, 22(3), 855. DOI: 10.3390/s22030855

[📄 View Paper (DOI: 10.3390/s22030855)](https://doi.org/10.3390/s22030855)

## Usage

```python
from opt.swarm_intelligence.pelican_optimizer import PelicanOptimizer
from opt.benchmark.functions import sphere

optimizer = PelicanOptimizer(
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
| `population_size` | `int` | `30` | Population size. |
| `max_iter` | `int` | `100` | Maximum iterations. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`pelican_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/pelican_optimizer.py)
:::
