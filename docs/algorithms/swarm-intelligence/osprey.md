# Osprey Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Osprey Optimization Algorithm (OOA).

## Algorithm Overview

This module implements the Osprey Optimization Algorithm, a nature-inspired
metaheuristic algorithm that mimics the hunting behavior of ospreys.

Ospreys are fish-eating birds of prey known for their remarkable hunting
skills. The algorithm simulates their hunting phases: position identification,
fish detection, and attack.

## Reference

> Dehghani, M., Trojovský, P., & Hubálovský, Š. (2023). Osprey optimization algorithm: A new bio-inspired metaheuristic algorithm for solving engineering optimization problems. Frontiers in Mechanical Engineering, 8, 1126450. DOI: 10.3389/fmech.2022.1126450

[📄 View Paper (DOI: 10.3389/fmech.2022.1126450)](https://doi.org/10.3389/fmech.2022.1126450)

## Usage

```python
from opt.swarm_intelligence.osprey_optimizer import OspreyOptimizer
from opt.benchmark.functions import sphere

optimizer = OspreyOptimizer(
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
View the implementation: [`osprey_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/osprey_optimizer.py)
:::
