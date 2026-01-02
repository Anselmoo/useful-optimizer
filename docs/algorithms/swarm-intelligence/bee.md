# Bee Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Bee Algorithm optimizer implementation.

## Algorithm Overview

This module provides an implementation of the Bee Algorithm optimizer.
The Bee Algorithm is a population-based optimization algorithm inspired
by the foraging behavior of honey bees. It is commonly used for solving
optimization problems.

The BeeAlgorithm class is the main class that implements the Bee Algorithm optimizer.
It takes an objective function, the dimensionality of the problem, and other optional
parameters as input. The search method runs the optimization process and returns the
best solution found and its corresponding fitness value.

Example usage:
    optimizer = BeeAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-2.768,
        upper_bound=+2.768,
        max_iter=4000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.swarm_intelligence.bee_algorithm import BeeAlgorithm
from opt.benchmark.functions import sphere

optimizer = BeeAlgorithm(
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
| `dim` | `int` | Required | Problem dimensionality. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `n_bees` | `int` | `50` | Number of bees in population. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `scout_bee` | `float` | `0.01` | Scout bee ratio. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`bee_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/bee_algorithm.py)
:::
