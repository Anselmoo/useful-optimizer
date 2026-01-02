# Nelder-Mead

<span class="badge badge-classical">Classical</span>

Nelder-Mead Optimizer.

## Algorithm Overview

This module implements the Nelder-Mead optimization algorithm. Nelder-Mead is a
derivative-free optimization method that uses only function evaluations (no gradients).
It works by maintaining a simplex of n+1 points in n-dimensional space and iteratively
replacing the worst point with a better one through reflection, expansion, contraction,
and shrinkage operations.

The Nelder-Mead method is particularly useful for:
- Functions where gradients are not available or difficult to compute
- Noisy functions
- Functions with discontinuities
- Black-box optimization problems

This implementation uses scipy's Nelder-Mead optimizer with multiple random restarts
to improve global optimization performance.

## Usage

```python
from opt.classical.nelder_mead import NelderMead
from opt.benchmark.functions import sphere

optimizer = NelderMead(
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
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`nelder_mead.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/nelder_mead.py)
:::
