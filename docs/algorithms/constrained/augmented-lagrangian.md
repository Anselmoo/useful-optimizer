# Augmented Lagrangian

<span class="badge badge-constrained">Constrained</span>

Augmented Lagrangian optimizer.

## Algorithm Overview

This module implements an optimizer based on the Augmented Lagrangian method. The
Augmented Lagrangian method is an optimization technique that combines the
advantages of both penalty and Lagrange multiplier methods. It is commonly used to
solve constrained optimization problems.

The `AugmentedLagrangian` class is the main class of this module. It takes an objective
function, lower and upper bounds of the search space, dimensionality of the search
space, and other optional parameters as input. It performs the search using the
Augmented Lagrangian method and returns the best solution found and its fitness value.

Example usage:
    optimizer = AugmentedLagrangian(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=+2.768,
        dim=2,
        max_iter=8000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Note:
    This module requires the `scipy` library to be installed.

## Usage

```python
from opt.constrained.augmented_lagrangian_method import AugmentedLagrangian
from opt.benchmark.functions import sphere

optimizer = AugmentedLagrangian(
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
| `func` | `Callable` | Required | Objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `c` | `float` | `1` | Algorithm-specific parameter |
| `lambda_` | `float` | `0.1` | Algorithm-specific parameter |
| `static_cost` | `float` | `10000000000.0` | Algorithm-specific parameter |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |

## See Also

- [Constrained Algorithms](/algorithms/constrained/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`augmented_lagrangian_method.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/constrained/augmented_lagrangian_method.py)
:::
