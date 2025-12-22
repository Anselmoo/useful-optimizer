# Black Widow Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Black Widow Optimization Algorithm.

## Algorithm Overview

Implementation based on:
Hayyolalam, V. & Kazem, A.A.P. (2020).
Black Widow Optimization Algorithm: A novel meta-heuristic approach
for solving engineering optimization problems.
Engineering Applications of Artificial Intelligence, 87, 103249.

The algorithm mimics the mating behavior of black widow spiders, including
cannibalistic behaviors where females may eat males after mating.

## Usage

```python
from opt.swarm_intelligence.black_widow import BlackWidowOptimizer
from opt.benchmark.functions import sphere

optimizer = BlackWidowOptimizer(
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
| `pp` | `float` | `_PP` | Algorithm-specific parameter |
| `cr` | `float` | `_CR` | Algorithm-specific parameter |
| `pm` | `float` | `_PM` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`black_widow.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/black_widow.py)
:::
