# Flower Pollination Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Flower Pollination Algorithm (FPA) implementation.

## Algorithm Overview

This module implements the Flower Pollination Algorithm, a nature-inspired
metaheuristic optimization algorithm based on the pollination process of
flowering plants.

## Reference

> Yang, X.-S. (2012). Flower pollination algorithm for global optimization. In Unconventional Computation and Natural Computation (pp. 240-249). Springer.

## Usage

```python
from opt.swarm_intelligence.flower_pollination import FlowerPollinationAlgorithm
from opt.benchmark.functions import sphere

optimizer = FlowerPollinationAlgorithm(
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
| `max_iter` | `int` | Required | Maximum iterations. |
| `population_size` | `int` | `25` | Population size. |
| `switch_probability` | `float` | `_SWITCH_PROBABILITY` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`flower_pollination.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/flower_pollination.py)
:::
