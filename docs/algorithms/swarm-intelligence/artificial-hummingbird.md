# Artificial Hummingbird Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Artificial Hummingbird Algorithm.

## Algorithm Overview

Implementation based on:
Zhao, W., Wang, L. & Mirjalili, S. (2022).
Artificial hummingbird algorithm: A new bio-inspired optimizer with
its engineering applications.
Computer Methods in Applied Mechanics and Engineering, 388, 114194.

The algorithm mimics the unique flight patterns and foraging behavior
of hummingbirds, known for their hovering capabilities.

## Usage

```python
from opt.swarm_intelligence.artificial_hummingbird import ArtificialHummingbirdAlgorithm
from opt.benchmark.functions import sphere

optimizer = ArtificialHummingbirdAlgorithm(
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
View the implementation: [`artificial_hummingbird.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/artificial_hummingbird.py)
:::
