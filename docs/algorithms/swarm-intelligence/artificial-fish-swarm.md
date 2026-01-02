# Artificial Fish Swarm

<span class="badge badge-swarm">Swarm Intelligence</span>

Artificial Fish Swarm Algorithm (AFSA).

## Algorithm Overview

This module implements the Artificial Fish Swarm Algorithm (AFSA). AFSA is a population
based optimization technique inspired by the social behavior of fishes. In their social
behavior, fish try to keep a balance between food consistency and crowding effect. This
behavior is modeled into a mathematical optimization technique in AFSA.

In AFSA, each fish represents a potential solution and the food consistency represents
the objective function to be optimized. Each fish tries to move towards better regions
of the search space based on its own experience and the experience of its neighbors.

AFSA has been used for various kinds of optimization problems including function
optimization, neural network training, fuzzy system control, and other areas of
engineering.

## Usage

```python
from opt.swarm_intelligence.artificial_fish_swarm_algorithm import ArtificialFishSwarm
from opt.benchmark.functions import sphere

optimizer = ArtificialFishSwarm(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
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
| `fish_swarm` | `int` | `50` | Number of fish in swarm. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `visual` | `int` | `1` | Visual distance parameter. |
| `step` | `float` | `0.1` | Step size parameter. |
| `try_number` | `int` | `3` | Number of attempts. |
| `epsilon` | `float` | `1e-09` | Small value for numerical stability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`artificial_fish_swarm_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/artificial_fish_swarm_algorithm.py)
:::
