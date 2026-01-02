# Reptile Search Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Reptile Search Algorithm (RSA) implementation.

## Algorithm Overview

This module implements the Reptile Search Algorithm, a nature-inspired
optimization algorithm based on the hunting behavior of crocodiles.

## Reference

> Abualigah, L., Abd Elaziz, M., Sumari, P., Geem, Z. W., & Gandomi, A. H. (2022). Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic optimizer. Expert Systems with Applications, 191, 116158.

## Usage

```python
from opt.swarm_intelligence.reptile_search import ReptileSearchAlgorithm
from opt.benchmark.functions import sphere

optimizer = ReptileSearchAlgorithm(
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
View the implementation: [`reptile_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/reptile_search.py)
:::
