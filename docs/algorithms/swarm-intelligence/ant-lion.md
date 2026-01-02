# Ant Lion Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Ant Lion Optimizer (ALO) Algorithm.

## Algorithm Overview

This module implements the Ant Lion Optimizer algorithm, a nature-inspired
metaheuristic based on the hunting mechanism of antlions.

Antlions dig cone-shaped pits in sand and wait for ants to fall in. When an ant
falls into the pit, the antlion throws sand outward to prevent escape. This hunting
mechanism is mathematically modeled for optimization.

## Reference

> Mirjalili, S. (2015). The Ant Lion Optimizer. Advances in Engineering Software, 83, 80-98. DOI: 10.1016/j.advengsoft.2015.01.010

[📄 View Paper (DOI: 10.1016/j.advengsoft.2015.01.010)](https://doi.org/10.1016/j.advengsoft.2015.01.010)

## Usage

```python
from opt.swarm_intelligence.ant_lion_optimizer import AntLionOptimizer
from opt.benchmark.functions import sphere

optimizer = AntLionOptimizer(
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
| `population_size` | `int` | `100` | Population size. |
| `track_history` | `bool` | `False` | Enable convergence history tracking for BBOB
        post-processing. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`ant_lion_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/ant_lion_optimizer.py)
:::
