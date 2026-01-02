# L-BFGS

<span class="badge badge-classical">Classical</span>

L-BFGS Optimizer.

## Algorithm Overview

This module implements the L-BFGS (Limited-memory BFGS) optimization algorithm.
L-BFGS is a quasi-Newton method that approximates the BFGS algorithm using a limited
amount of computer memory. It's particularly useful for large-scale optimization
problems where storing the full inverse Hessian approximation would be prohibitive.

L-BFGS maintains only a few vectors that represent the approximation implicitly,
making it much more memory-efficient than full BFGS while retaining similar
convergence properties.

This implementation uses scipy's L-BFGS-B optimizer with multiple random restarts
to improve global optimization performance.

## Usage

```python
from opt.classical.lbfgs import LBFGS
from opt.benchmark.functions import sphere

optimizer = LBFGS(
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
View the implementation: [`lbfgs.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/lbfgs.py)
:::
