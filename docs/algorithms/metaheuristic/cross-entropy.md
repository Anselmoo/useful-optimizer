# Cross Entropy Method

<span class="badge badge-metaheuristic">Metaheuristic</span>

Cross-Entropy Method (CEM) optimizer implementation.

## Algorithm Overview

This module provides an implementation of the Cross-Entropy Method (CEM) optimizer. The
CEM algorithm is a stochastic optimization method that is particularly effective for
solving problems with continuous search spaces.

The CrossEntropyMethod class is the main class of this module and serves as the
optimizer. It takes an objective function, lower and upper bounds of the search space,
dimensionality of the search space, and other optional parameters as input. It uses the
CEM algorithm to find the optimal solution for the given objective function within the
specified search space.

Example usage:
    optimizer = CrossEntropyMethod(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.metaheuristic.cross_entropy_method import CrossEntropyMethod
from opt.benchmark.functions import sphere

optimizer = CrossEntropyMethod(
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
| `population_size` | `int` | `100` | Number of samples per iteration. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `elite_frac` | `float` | `0.2` | Fraction of samples to use as elite set. |
| `noise_decay` | `float` | `0.99` | Covariance decay factor to maintain exploration. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cross_entropy_method.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/cross_entropy_method.py)
:::
