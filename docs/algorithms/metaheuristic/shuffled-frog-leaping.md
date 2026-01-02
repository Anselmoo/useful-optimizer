# Shuffled Frog Leaping Algorithm

<span class="badge badge-metaheuristic">Metaheuristic</span>

Shuffled Frog Leaping Algorithm (SFLA) optimizer implementation.

## Algorithm Overview

This module provides an implementation of the Shuffled Frog Leaping Algorithm (SFLA)
optimizer. The SFLA is a population-based optimization algorithm inspired by the
behavior of frogs in a pond. It is used to solve optimization problems by iteratively
improving a population of candidate solutions.

The algorithm works by maintaining a population of frogs, where each frog represents a
candidate solution. In each iteration, the frogs are shuffled and leaped towards the
mean position of the best frogs. This process helps explore the search space and
converge towards the optimal solution.

This module defines the `ShuffledFrogLeapingAlgorithm` class, which is responsible for
executing the optimization process. The class takes an objective function, lower and
upper bounds of the search space, dimensionality of the search space, population size,
maximum number of iterations, and other optional parameters as input.

Example usage:
    optimizer = ShuffledFrogLeapingAlgorithm(
        func=shifted_ackley, lower_bound=-32.768, upper_bound=+32.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Fitness value: {best_fitness}")

## Usage

```python
from opt.metaheuristic.shuffled_frog_leaping_algorithm import ShuffledFrogLeapingAlgorithm
from opt.benchmark.functions import sphere

optimizer = ShuffledFrogLeapingAlgorithm(
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
| `population_size` | `int` | `100` | Total number of frogs. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `cut` | `int` | `2` | Number of memeplexes (population divisions). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`shuffled_frog_leaping_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/shuffled_frog_leaping_algorithm.py)
:::
