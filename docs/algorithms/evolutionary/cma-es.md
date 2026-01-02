# CMA-ES

<span class="badge badge-evolutionary">Evolutionary</span>

Covariance Matrix Adaptation Evolution Strategy.

## Algorithm Overview

This module implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm,
which is a derivative-free optimization method that uses an evolutionary strategy to search for
the optimal solution. It adapts the covariance matrix of the multivariate Gaussian distribution
to guide the search towards promising regions of the search space.

The CMA-ES algorithm is implemented in the `CMAESAlgorithm` class, which inherits from the
`AbstractOptimizer` class. The `CMAESAlgorithm` class provides a `search` method that runs the
CMA-ES algorithm to search for the optimal solution.

Example usage:
    optimizer = CMAESAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-12.768,
        upper_bound=12.768,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

## Usage

```python
from opt.evolutionary.cma_es import CMAESAlgorithm
from opt.benchmark.functions import sphere

optimizer = CMAESAlgorithm(
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
| `dim` | `int` | Required | Problem dimensionality. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `population_size` | `int` | `100` | Number of offspring per generation (λ). |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `sigma_init` | `float` | `0.5` | Initial global step-size controlling search spread. |
| `epsilon` | `float` | `1e-09` | Minimum step-size threshold to prevent numerical instability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cma_es.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/evolutionary/cma_es.py)
:::
