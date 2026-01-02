# Eagle Strategy

<span class="badge badge-metaheuristic">Metaheuristic</span>

Eagle Strategy Optimization Algorithm.

## Algorithm Overview

This module implements the Eagle Strategy (ES) optimization algorithm. ES is a
metaheuristic optimization algorithm inspired by the hunting behavior of eagles.
The algorithm mimics the way eagles soar, glide, and swoop down to catch their prey.

In ES, each eagle represents a potential solution, and the objective function
determines the quality of the solutions. The eagles try to update their positions by
mimicking the hunting behavior of eagles, which includes soaring, gliding, and swooping.

ES has been used for various kinds of optimization problems including function
optimization, neural network training, and other areas of engineering.

## Usage

```python
from opt.metaheuristic.eagle_strategy import EagleStrategy
from opt.benchmark.functions import sphere

optimizer = EagleStrategy(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Number of eagles. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`eagle_strategy.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/eagle_strategy.py)
:::
