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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `population_size` | `int` | `100` | The number of individuals in the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `init_temperature` | `float` | `1000` | The initial temperature. |
| `stopping_temperature` | `float` | `1e-08` | The stopping temperature. |
| `cooling_rate` | `float` | `0.99` | The cooling rate. |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`simulated_annealing.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/simulated_annealing.py)
:::
