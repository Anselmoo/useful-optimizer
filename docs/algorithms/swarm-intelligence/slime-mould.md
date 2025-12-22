# Slime Mould Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Slime Mould Algorithm (SMA) implementation.

## Algorithm Overview

This module implements the Slime Mould Algorithm, a nature-inspired
optimization algorithm based on the oscillation mode of slime mould
in nature during foraging.

## Reference

> Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020). Slime mould algorithm: A new method for stochastic optimization. Future Generation Computer Systems, 111, 300-323.

## Usage

```python
from opt.swarm_intelligence.slime_mould import SlimeMouldAlgorithm
from opt.benchmark.functions import sphere

optimizer = SlimeMouldAlgorithm(
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
| `func` | `Callable` | Required | Objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `max_iter` | `int` | Required | Maximum number of iterations |
| `population_size` | `int` | `30` | Number of individuals in population |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`slime_mould.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/slime_mould.py)
:::
