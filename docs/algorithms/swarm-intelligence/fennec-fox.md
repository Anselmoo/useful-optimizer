# Fennec Fox Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Fennec Fox Optimization (FFO) Algorithm.

## Algorithm Overview

This module implements the Fennec Fox Optimization algorithm, a nature-inspired
metaheuristic based on the survival behaviors of fennec foxes in the desert.

Fennec foxes use two main strategies: seeking prey and escaping from predators.
Their large ears help them detect prey underground and predators from afar.

## Reference

> Trojovská, E., Dehghani, M., & Trojovský, P. (2023). Fennec Fox Optimization: A New Nature-Inspired Optimization Algorithm. IEEE Access, 10, 84417-84443. DOI: 10.1109/ACCESS.2022.3197745

[📄 View Paper (DOI: 10.1109/ACCESS.2022.3197745)](https://doi.org/10.1109/ACCESS.2022.3197745)

## Usage

```python
from opt.swarm_intelligence.fennec_fox import FennecFoxOptimizer
from opt.benchmark.functions import sphere

optimizer = FennecFoxOptimizer(
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
View the implementation: [`fennec_fox.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/fennec_fox.py)
:::
