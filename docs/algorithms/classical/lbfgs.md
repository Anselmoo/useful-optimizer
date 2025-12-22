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
| `func` | `Callable` | Required | The objective function to be optimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `num_restarts` | `int` | `25` | Number of random restarts. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`lbfgs.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/lbfgs.py)
:::
