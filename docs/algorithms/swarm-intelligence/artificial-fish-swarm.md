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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `fish_swarm` | `int` | `50` | The number of fish in the swarm (default: 50). |
| `max_iter` | `int` | `1000` | The maximum number of iterations (default: 1000). |
| `visual` | `int` | `1` | The visual range of each fish (default: 1). |
| `step` | `float` | `0.1` | The step size for fish movement (default: 0. |
| `try_number` | `int` | `3` | The number of attempts for prey behavior (default: 3). |
| `epsilon` | `float` | `1e-09` | A small value added to avoid division by zero (default: 1e-9). |
| `seed` | `int  \|  None` | `None` | The seed for the random number generator (default: None). |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`artificial_fish_swarm_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/artificial_fish_swarm_algorithm.py)
:::
