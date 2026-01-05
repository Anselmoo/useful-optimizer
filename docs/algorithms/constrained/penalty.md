# Penalty Method Optimizer

<span class="badge badge-constrained">Constrained</span>

Penalty Method Optimizer.

## Algorithm Overview

This module implements the Penalty Method for constrained optimization,
transforming constrained problems into unconstrained ones.

The algorithm adds penalty terms for constraint violations to the objective
function, with increasing penalty coefficients over iterations.

## Reference

> Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer. Chapter 17: Penalty and Augmented Lagrangian Methods.

## Usage

```python
from opt.constrained.penalty_method import PenaltyMethodOptimizer
from opt.benchmark.functions import sphere

optimizer = PenaltyMethodOptimizer(
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
| `eq_constraints` | `list[Callable]  \|  None` | `None` | List of
        equality constraints in form $h(x) = 0$. |
| `max_iter` | `int` | `100` | Maximum outer iterations. |
| `initial_penalty` | `float` | `1.0` | Starting penalty coefficient ρ₀. |
| `penalty_growth` | `float` | `2.0` | Penalty growth factor gamma > 1. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Constrained Algorithms](/algorithms/constrained/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`penalty_method.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/constrained/penalty_method.py)
:::
