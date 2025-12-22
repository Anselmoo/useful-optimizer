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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `dim` | `int` | Required | The dimensionality of the problem. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `n_bats` | `int` | Required | The number of bats in the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `loudness` | `float` | `0.5` | The initial loudness of the bats. |
| `pulse_rate` | `float` | `0.9` | The pulse rate of the bats. |
| `freq_min` | `float` | `0` | The minimum frequency of the bats. |
| `freq_max` | `float` | `2` | The maximum frequency of the bats. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`bat_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/bat_algorithm.py)
:::
