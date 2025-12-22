# Zebra Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Zebra Optimization Algorithm (ZOA).

## Algorithm Overview

This module implements the Zebra Optimization Algorithm, a nature-inspired
metaheuristic based on the foraging and defense behaviors of zebras.

Zebras exhibit two main behaviors: foraging (searching for food and water)
and defense against predators through collective movement and vigilance.

## Reference

> Trojovská, E., Dehghani, M., & Trojovský, P. (2022). Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm for Solving Optimization Problems. IEEE Access, 10, 49445-49473. DOI: 10.1109/ACCESS.2022.3172789

[📄 View Paper (DOI: 10.1109/ACCESS.2022.3172789)](https://doi.org/10.1109/ACCESS.2022.3172789)

## Usage

```python
from opt.swarm_intelligence.zebra_optimizer import ZebraOptimizer
from opt.benchmark.functions import sphere

optimizer = ZebraOptimizer(
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
| `population_size` | `int` | `30` | Number of individuals in population |
| `max_iter` | `int` | `100` | Maximum number of iterations |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`zebra_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/zebra_optimizer.py)
:::
