# Stochastic Diffusion Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Stochastic Diffusion Search optimizer.

## Algorithm Overview

This module implements the Stochastic Diffusion Search optimizer, which is an
optimization algorithm that uses a population of agents to explore the search space and
find the optimal solution for a given objective function.

The main class in this module is `StochasticDiffusionSearch`, which represents the
optimizer. It takes the objective function, lower and upper bounds of the search space,
dimensionality of the search space, population size, maximum number of iterations,
and seed for the random number generator as input parameters.

The optimizer works by initializing a population of agents, where each agent has a
position in the search space and a score based on the objective function. The algorithm
then iteratively performs a test phase and a diffusion phase to update the positions of
the agents. After the specified number of iterations, the algorithm returns the best
solution found and its corresponding score.

Example usage:
    optimizer = StochasticDiffusionSearch(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

## Usage

```python
from opt.metaheuristic.stochastic_diffusion_search import StochasticDiffusionSearch
from opt.benchmark.functions import sphere

optimizer = StochasticDiffusionSearch(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Number of agents. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`stochastic_diffusion_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/stochastic_diffusion_search.py)
:::
