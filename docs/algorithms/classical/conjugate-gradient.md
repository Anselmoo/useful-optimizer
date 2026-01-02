# Conjugate Gradient

<span class="badge badge-classical">Classical</span>

Conjugate Gradient Optimizer.

## Algorithm Overview

This module implements the Conjugate Gradient optimization algorithm. The Conjugate
Gradient method is an algorithm for the numerical solution of systems of linear
equations whose matrix is positive-definite. For general optimization, it's used
as an iterative method for solving unconstrained optimization problems.

The method works by:
1. Computing the gradient at the current point
2. Determining a conjugate direction (orthogonal in a specific sense)
3. Performing a line search along this direction
4. Updating the position and computing a new conjugate direction

The conjugate gradient method has the property that it converges in at most n steps
for a quadratic function in n dimensions, making it particularly effective for
quadratic and near-quadratic problems.

This implementation uses scipy's CG optimizer with multiple random restarts
to improve global optimization performance.

## Usage

```python
from opt.classical.conjugate_gradient import ConjugateGradient
from opt.benchmark.functions import sphere

optimizer = ConjugateGradient(
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
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `max_iter` | `int` | `1000` | Maximum iterations per restart. |
| `num_restarts` | `int` | `10` | Number of random restarts for multistart strategy. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`conjugate_gradient.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/conjugate_gradient.py)
:::
