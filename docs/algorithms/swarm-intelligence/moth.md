# Moth Search Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Moth Search Algorithm.

## Algorithm Overview

Implementation based on:
Wang, G.G. (2018).
Moth search algorithm: a bio-inspired metaheuristic algorithm for
global optimization problems.
Memetic Computing, 10(2), 151-164.

The algorithm mimics the phototaxis behavior of moths toward light sources
(Lévy flights) and the spiral flying path around the flame.

## Usage

```python
from opt.swarm_intelligence.moth_search import MothSearchAlgorithm
from opt.benchmark.functions import sphere

optimizer = MothSearchAlgorithm(
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
| `path_finder_ratio` | `float` | `0.5` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`moth_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/moth_search.py)
:::
