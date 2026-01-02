# Orca Predator Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Orca Predator Algorithm.

## Algorithm Overview

Implementation based on:
Jiang, N., Wang, W., Yin, Z., Li, Y. & Zhao, S. (2022).
Orca Predation Algorithm: A new bio-inspired optimizer
for engineering optimization problems.
Expert Systems with Applications, 209, 118321.

The algorithm mimics the hunting strategies of orca whales,
combining carousel feeding and wave-wash feeding techniques.

## Usage

```python
from opt.swarm_intelligence.orca_predator import OrcaPredatorAlgorithm
from opt.benchmark.functions import sphere

optimizer = OrcaPredatorAlgorithm(
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
| `population_size` | `int` | `30` | Population size. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`orca_predator.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/orca_predator.py)
:::
