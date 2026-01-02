# Barnacles Mating Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Barnacles Mating Optimizer.

## Algorithm Overview

Implementation based on:
Sulaiman, M.H., Mustaffa, Z., Saari, M.M. & Daniyal, H. (2020).
Barnacles Mating Optimizer: A new bio-inspired algorithm for solving
engineering optimization problems.
Engineering Applications of Artificial Intelligence, 87, 103330.

The algorithm mimics the mating behavior of barnacles, where sessile
creatures must extend their reproductive organs to reach nearby mates.

## Usage

```python
from opt.swarm_intelligence.barnacles_mating import BarnaclesMatingOptimizer
from opt.benchmark.functions import sphere

optimizer = BarnaclesMatingOptimizer(
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
| `max_iter` | `int` | Required | Maximum iterations. |
| `population_size` | `int` | `30` | Population size. |
| `pl` | `int` | `_PL` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`barnacles_mating.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/barnacles_mating.py)
:::
