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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `population_size` | `int` | `100` | The number of solutions in each generation. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `sigma_init` | `float` | `0.5` | The initial step size. |
| `epsilon` | `float` | `1e-09` | A small value to prevent the step size from becoming too small. |
| `seed` | `int  \|  None` | `None` | The random seed. |

## See Also

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cma_es.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/evolutionary/cma_es.py)
:::
