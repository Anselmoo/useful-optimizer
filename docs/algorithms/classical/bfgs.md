# BFGS

<span class="badge badge-classical">Classical</span>

BFGS Optimizer.

## Algorithm Overview

This module implements the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm.
BFGS is a quasi-Newton method that approximates Newton's method by using an approximation
to the inverse Hessian matrix. It's particularly effective for smooth optimization problems
and typically converges faster than first-order methods.

BFGS builds up an approximation to the inverse Hessian matrix using gradient information
from previous iterations. This makes it more efficient than computing the actual Hessian
while still providing second-order convergence properties.

This implementation uses scipy's BFGS optimizer with multiple random restarts to improve
global optimization performance.

## Usage

```python
from opt.classical.bfgs import BFGS
from opt.benchmark.functions import sphere

optimizer = BFGS(
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
| `num_restarts` | `int` | `25` | Number of random restarts for multistart strategy. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`bfgs.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/bfgs.py)
:::
