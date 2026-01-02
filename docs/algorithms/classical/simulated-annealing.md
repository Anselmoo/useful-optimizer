# Simulated Annealing

<span class="badge badge-classical">Classical</span>

Simulated Annealing optimizer.

## Algorithm Overview

This module provides an implementation of the Simulated Annealing optimization
algorithm. Simulated Annealing is a metaheuristic optimization algorithm that is
inspired by the annealing process in metallurgy. It is used to find the global minimum
of a given objective function in a search space.

## Usage

```python
from opt.classical.simulated_annealing import SimulatedAnnealing
from opt.benchmark.functions import sphere

optimizer = SimulatedAnnealing(
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
| `population_size` | `int` | `100` | Number of independent runs. |
| `max_iter` | `int` | `1000` | Maximum iterations per run. |
| `init_temperature` | `float` | `1000` | Initial temperature. |
| `stopping_temperature` | `float` | `1e-08` | Temperature stopping criterion. |
| `cooling_rate` | `float` | `0.99` | Geometric cooling factor ($0 < \alpha < 1$). |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`simulated_annealing.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/simulated_annealing.py)
:::
