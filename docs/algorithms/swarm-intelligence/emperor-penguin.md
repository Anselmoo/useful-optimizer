# Emperor Penguin Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Emperor Penguin Optimizer (EPO) implementation.

## Algorithm Overview

This module implements the Emperor Penguin Optimizer, a nature-inspired
metaheuristic based on the huddling behavior of emperor penguins
to survive the harsh Antarctic winter.

## Reference

> Dhiman, G., & Kumar, V. (2018). Emperor penguin optimizer: A bio-inspired algorithm for engineering problems. Knowledge-Based Systems, 159, 20-50.

## Usage

```python
from opt.swarm_intelligence.emperor_penguin import EmperorPenguinOptimizer
from opt.benchmark.functions import sphere

optimizer = EmperorPenguinOptimizer(
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
View the implementation: [`emperor_penguin.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/emperor_penguin.py)
:::
