# Wild Horse Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Wild Horse Optimizer.

## Algorithm Overview

Implementation based on:
Naruei, I. & Keynia, F. (2022).
Wild Horse Optimizer: A new meta-heuristic algorithm for solving
engineering optimization problems.
Engineering with Computers, 38(4), 3025-3056.

The algorithm mimics the social behavior of wild horses including
grazing, fighting, and herd dynamics.

## Usage

```python
from opt.swarm_intelligence.wild_horse import WildHorseOptimizer
from opt.benchmark.functions import sphere

optimizer = WildHorseOptimizer(
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
| `n_groups` | `int` | `5` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`wild_horse.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/wild_horse.py)
:::
