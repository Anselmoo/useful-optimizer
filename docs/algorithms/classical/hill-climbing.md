# Hill Climbing

<span class="badge badge-classical">Classical</span>

Hill Climbing optimizer.

## Algorithm Overview

This module implements the Hill Climbing optimizer, which performs a hill climbing
search to find the optimal solution for a given function within the specified bounds.

The HillClimbing class is the main class that implements the optimizer. It takes the
objective function, lower and upper bounds of the search space, dimensionality of the
search space, and other optional parameters as input. The search method performs the
hill climbing search and returns the optimal solution and its corresponding score.

Example usage:
    optimizer = HillClimbing(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        max_iter=5000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.classical.hill_climbing import HillClimbing
from opt.benchmark.functions import sphere

optimizer = HillClimbing(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `initial_step_sizes` | `float` | `1.0` | Initial step size for all dimensions. |
| `acceleration` | `float` | `1.2` | Factor for step size adaptation. |
| `epsilon` | `float` | `1e-06` | Convergence threshold for fitness change. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`hill_climbing.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/hill_climbing.py)
:::
