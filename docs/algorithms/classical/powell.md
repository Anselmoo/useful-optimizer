# Powell's Method

<span class="badge badge-classical">Classical</span>

Powell Optimizer.

## Algorithm Overview

This module implements Powell's optimization algorithm. Powell's method is a
derivative-free optimization algorithm that performs sequential one-dimensional
minimizations along coordinate directions and then updates the search directions
based on the progress made.

Powell's method works by:
1. Starting with a set of linearly independent directions (usually coordinate axes)
2. Performing line searches along each direction
3. Replacing one of the directions with the overall direction of progress
4. Repeating until convergence

The method is particularly effective for functions that are not too irregular
and can handle functions where gradients are not available.

This implementation uses scipy's Powell optimizer with multiple random restarts
to improve global optimization performance.

## Usage

```python
from opt.classical.powell import Powell
from opt.benchmark.functions import sphere

optimizer = Powell(
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
| `num_restarts` | `int` | `10` | Number of random restarts. |
| `seed` | `int  \|  None` | `None` | Random seed for BBOB reproducibility. |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`powell.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/powell.py)
:::
