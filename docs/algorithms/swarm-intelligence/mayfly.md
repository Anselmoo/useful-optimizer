# Mayfly Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Mayfly Optimization Algorithm.

## Algorithm Overview

Implementation based on:
Zervoudakis, K. & Tsafarakis, S. (2020).
A mayfly optimization algorithm.
Computers & Industrial Engineering, 145, 106559.

The algorithm mimics the mating behavior of mayflies, including nuptial
dances performed by males to attract females and the swarm dynamics of
both male and female mayflies.

## Usage

```python
from opt.swarm_intelligence.mayfly_optimizer import MayflyOptimizer
from opt.benchmark.functions import sphere

optimizer = MayflyOptimizer(
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
| `a1` | `float` | `_A1` | Algorithm-specific parameter |
| `a2` | `float` | `_A2` | Algorithm-specific parameter |
| `a3` | `float` | `_A3` | Algorithm-specific parameter |
| `beta` | `float` | `_BETA` | Algorithm-specific parameter |
| `dance` | `float` | `_DANCE` | Algorithm-specific parameter |
| `fl` | `float` | `_FL` | Algorithm-specific parameter |
| `g` | `float` | `_G` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`mayfly_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/mayfly_optimizer.py)
:::
