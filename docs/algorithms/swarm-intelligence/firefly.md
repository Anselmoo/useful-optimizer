# Firefly Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Firefly Algorithm implementation.

## Algorithm Overview

This module provides an implementation of the Firefly Algorithm optimization algorithm.
The Firefly Algorithm is a metaheuristic optimization algorithm inspired by the
flashing behavior of fireflies. It is commonly used to solve optimization problems by
simulating the behavior of fireflies in attracting each other.

The algorithm works by representing potential solutions as fireflies in a search space.
Each firefly's brightness is determined by its fitness value, with brighter fireflies
representing better solutions. Fireflies move towards brighter fireflies in the search
space, and their movements are influenced by attractiveness and light absorption
coefficients.

This implementation provides a class called FireflyAlgorithm, which can be used to
perform optimization using the Firefly Algorithm. The class takes an objective
function, lower and upper bounds of the search space, dimensionality of the search
space, and other optional parameters. The search method of the class runs the
Firefly Algorithm optimization and returns the best solution found.

Example usage:
    optimizer = FireflyAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=32.768,
        population_size=100,
        max_iter=1000,
        alpha=0.5,
        beta_0=1,
        gamma=1,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.swarm_intelligence.firefly_algorithm import FireflyAlgorithm
from opt.benchmark.functions import sphere

optimizer = FireflyAlgorithm(
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
| `population_size` | `int` | `100` | Number of fireflies in the population. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `alpha` | `float` | `0.5` | Randomization parameter controlling step size
        of random movement. |
| `beta_0` | `float` | `1` | Attractiveness coefficient at distance r=0. |
| `gamma` | `float` | `1` | Light absorption coefficient. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`firefly_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/firefly_algorithm.py)
:::
