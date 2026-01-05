# Bat Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Bat Algorithm optimization algorithm.

## Algorithm Overview

This module implements the Bat Algorithm optimization algorithm. The Bat Algorithm is a
metaheuristic algorithm inspired by the echolocation behavior of bats. It is commonly
used for solving optimization problems.

The BatAlgorithm class provides an implementation of the Bat Algorithm optimization
algorithm. It takes an objective function, the dimensionality of the problem, the
search space bounds, the number of bats in the population, and other optional
parameters. The search method runs the Bat Algorithm optimization and returns the
best solution found.

## Usage

```python
from opt.swarm_intelligence.bat_algorithm import BatAlgorithm
from opt.benchmark.functions import sphere

optimizer = BatAlgorithm(
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
| `dim` | `int` | Required | Problem dimensionality. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `n_bats` | `int` | Required | Number of bats in the population. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `loudness` | `float` | `0.5` | Initial loudness parameter (0-1). |
| `pulse_rate` | `float` | `0.9` | Pulse emission rate (0-1). |
| `freq_min` | `float` | `0` | Minimum frequency for velocity updates. |
| `freq_max` | `float` | `2` | Maximum frequency for velocity updates. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`bat_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/bat_algorithm.py)
:::
