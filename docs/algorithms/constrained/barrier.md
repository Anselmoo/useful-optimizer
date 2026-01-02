# Barrier Method Optimizer

<span class="badge badge-constrained">Constrained</span>

Barrier Method (Interior Point) Optimizer.

## Algorithm Overview

This module implements the Barrier Method for constrained optimization,
also known as the Interior Point Method.

The algorithm uses logarithmic barrier functions to keep solutions strictly
inside the feasible region while optimizing the objective.

## Reference

> Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press. Chapter 11: Interior-Point Methods.

## Usage

```python
from opt.constrained.barrier_method import BarrierMethodOptimizer
from opt.benchmark.functions import sphere

optimizer = BarrierMethodOptimizer(
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
| `constraints` | `list[Callable]  \|  None` | `None` | List of
        inequality constraints in form $g(x) \leq 0$. |
| `max_iter` | `int` | `100` | Maximum outer iterations. |
| `initial_mu` | `float` | `10.0` | Starting barrier coefficient. |
| `mu_reduction` | `float` | `0.5` | Barrier reduction factor β (0 < β < 1). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Constrained Algorithms](/algorithms/constrained/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`barrier_method.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/constrained/barrier_method.py)
:::
