# Sequential Quadratic Programming

<span class="badge badge-constrained">Constrained</span>

Sequential Quadratic Programming Optimizer.

## Algorithm Overview

This module implements the Sequential Quadratic Programming (SQP) algorithm,
a powerful method for solving nonlinear constrained optimization problems.

The algorithm iteratively solves quadratic programming subproblems to
approximate the original nonlinear problem.

## Reference

> Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.). Springer. Chapter 18: Sequential Quadratic Programming.

## Usage

```python
from opt.constrained.sequential_quadratic_programming import SequentialQuadraticProgramming
from opt.benchmark.functions import sphere

optimizer = SequentialQuadraticProgramming(
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
| `constraints` | `list[Callable]  \|  None` | `None` | Algorithm-specific parameter |
| `eq_constraints` | `list[Callable]  \|  None` | `None` | Algorithm-specific parameter |
| `max_iter` | `int` | `100` | Maximum number of iterations |
| `tol` | `float` | `1e-06` | Algorithm-specific parameter |

## See Also

- [Constrained Algorithms](/algorithms/constrained/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`sequential_quadratic_programming.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/constrained/sequential_quadratic_programming.py)
:::
