# Successive Linear Programming

<span class="badge badge-constrained">Constrained</span>

Successive Linear Programming optimization algorithm.

## Algorithm Overview

!!! warning

    This module is still under development and is not yet ready for use.

This module implements the Successive Linear Programming optimization algorithm. The
algorithm performs a search for the optimal solution by iteratively updating a
population of individuals. At each iteration, it computes the gradient of the objective
function for each individual and uses linear programming to find a new solution that
improves the objective function value. The process continues until the maximum number
of iterations is reached.

The SuccessiveLinearProgramming class is the main class that implements the algorithm.
It inherits from the AbstractOptimizer class and overrides the search() and gradient()
methods.

## Usage

```python
from opt.constrained.successive_linear_programming import SuccessiveLinearProgramming
from opt.benchmark.functions import sphere

optimizer = SuccessiveLinearProgramming(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
    population_size=50,
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
| `max_iter` | `int` | `1000` | Maximum SLP iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Population size for gradient estimation via
        finite differences. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Constrained Algorithms](/algorithms/constrained/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`successive_linear_programming.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/constrained/successive_linear_programming.py)
:::
