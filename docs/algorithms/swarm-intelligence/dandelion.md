# Dandelion Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Dandelion Optimizer (DO).

## Algorithm Overview

This module implements the Dandelion Optimizer, a bio-inspired metaheuristic
algorithm based on the seed dispersal behavior of dandelions.

Dandelions disperse seeds through wind, with seeds traveling in different
patterns depending on wind conditions - from gentle floating to long-distance
travel.

## Reference

> Zhao, S., Zhang, T., Ma, S., & Chen, M. (2022). Dandelion Optimizer: A nature-inspired metaheuristic algorithm for engineering applications. Engineering Applications of Artificial Intelligence, 114, 105075. DOI: 10.1016/j.engappai.2022.105075

[📄 View Paper (DOI: 10.1016/j.engappai.2022.105075)](https://doi.org/10.1016/j.engappai.2022.105075)

## Usage

```python
from opt.swarm_intelligence.dandelion_optimizer import DandelionOptimizer
from opt.benchmark.functions import sphere

optimizer = DandelionOptimizer(
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
| `population_size` | `int` | `30` | Population size. |
| `max_iter` | `int` | `100` | Maximum iterations. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`dandelion_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/dandelion_optimizer.py)
:::
