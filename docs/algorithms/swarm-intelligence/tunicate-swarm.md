# Tunicate Swarm Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Tunicate Swarm Algorithm (TSA) implementation.

## Algorithm Overview

This module implements the Tunicate Swarm Algorithm, a bio-inspired
optimization algorithm based on the swarm behavior of tunicates
(sea squirts) during navigation and foraging.

## Reference

> Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate Swarm Algorithm: A new bio-inspired based metaheuristic paradigm for global optimization. Engineering Applications of Artificial Intelligence, 90, 103541.

## Usage

```python
from opt.swarm_intelligence.tunicate_swarm import TunicateSwarmAlgorithm
from opt.benchmark.functions import sphere

optimizer = TunicateSwarmAlgorithm(
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

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`tunicate_swarm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/tunicate_swarm.py)
:::
