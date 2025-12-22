# Glowworm Swarm Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Glowworm Swarm Optimization (GSO) algorithm.

## Algorithm Overview

This module implements the Glowworm Swarm Optimization (GSO) algorithm as an optimizer.
GSO is a population-based optimization algorithm inspired by the behavior of glowworms.
It is commonly used to solve optimization problems.

The GlowwormSwarmOptimization class provides an implementation of the GSO algorithm. It
takes an objective function, lower and upper bounds of the search space, dimensionality
of the search space, and other optional parameters as input. The algorithm searches for
the best solution within the given search space by iteratively updating the positions of
glowworms based on their luciferin levels and neighboring glowworms.

Usage:
    optimizer = GlowwormSwarmOptimization(
        func=shifted_ackley, dim=2, lower_bound=-32.768, upper_bound=+32.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.swarm_intelligence.glowworm_swarm_optimization import GlowwormSwarmOptimization
from opt.benchmark.functions import sphere

optimizer = GlowwormSwarmOptimization(
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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `population_size` | `int` | `100` | The number of glowworms in the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `luciferin_decay` | `float` | `0.1` | The decay rate of luciferin. |
| `randomness` | `float` | `0.5` | The randomness factor for glowworm movement. |
| `step_size` | `float` | `0.01` | The step size for glowworm movement. |
| `seed` | `int  \|  None` | `None` | The seed for the random number generator. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`glowworm_swarm_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/glowworm_swarm_optimization.py)
:::
