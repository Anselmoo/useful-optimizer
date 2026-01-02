# Starling Murmuration Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Starling Murmuration Optimizer (SMO).

## Algorithm Overview

This module implements the Starling Murmuration Optimizer, a swarm intelligence
algorithm inspired by the collective behavior of starlings during murmuration.

Murmurations are the stunning aerial displays created when thousands of
starlings fly together, creating complex patterns while maintaining cohesion.

## Reference

> Zamani, H., Nadimi-Shahraki, M. H., & Gandomi, A. H. (2022). Starling murmuration optimizer: A novel bio-inspired algorithm for global and engineering optimization. Computer Methods in Applied Mechanics and Engineering, 392, 114616. DOI: 10.1016/j.cma.2022.114616

[📄 View Paper (DOI: 10.1016/j.cma.2022.114616)](https://doi.org/10.1016/j.cma.2022.114616)

## Usage

```python
from opt.swarm_intelligence.starling_murmuration import StarlingMurmurationOptimizer
from opt.benchmark.functions import sphere

optimizer = StarlingMurmurationOptimizer(
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
View the implementation: [`starling_murmuration.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/starling_murmuration.py)
:::
