# Spotted Hyena Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Spotted Hyena Optimizer (SHO) implementation.

## Algorithm Overview

This module implements the Spotted Hyena Optimizer, a nature-inspired
metaheuristic algorithm based on the social behavior and hunting
strategies of spotted hyenas.

## Reference

> Dhiman, G., & Kumar, V. (2017). Spotted hyena optimizer: A novel bio-inspired based metaheuristic technique for engineering applications. Advances in Engineering Software, 114, 48-70.

## Usage

```python
from opt.swarm_intelligence.spotted_hyena import SpottedHyenaOptimizer
from opt.benchmark.functions import sphere

optimizer = SpottedHyenaOptimizer(
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
View the implementation: [`spotted_hyena.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/spotted_hyena.py)
:::
