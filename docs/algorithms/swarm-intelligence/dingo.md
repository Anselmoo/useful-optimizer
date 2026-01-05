# Dingo Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Dingo Optimizer.

## Algorithm Overview

Implementation based on:
Peraza-Vázquez, H., Peña-Delgado, A.F., Echavarría-Castillo, G.,
Morales-Cepeda, A.B., Velasco-Álvarez, J. & Ruiz-Perez, F. (2021).
A Bio-Inspired Method for Engineering Design Optimization Inspired
by Dingoes Hunting Strategies.
Mathematical Problems in Engineering, 2021, 9107547.

The algorithm mimics the hunting strategies of dingoes, including
pack hunting, persecution, and attacking behavior.

## Usage

```python
from opt.swarm_intelligence.dingo_optimizer import DingoOptimizer
from opt.benchmark.functions import sphere

optimizer = DingoOptimizer(
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
| `survival_rate` | `float` | `_SURVIVAL_RATE` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`dingo_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/dingo_optimizer.py)
:::
