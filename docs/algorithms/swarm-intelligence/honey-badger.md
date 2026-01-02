# Honey Badger Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Honey Badger Algorithm.

## Algorithm Overview

Implementation based on:
Hashim, F.A., Houssein, E.H., Hussain, K., Mabrouk, M.S. & Al-Atabany, W. (2022).
Honey Badger Algorithm: New metaheuristic algorithm for solving optimization
problems.
Mathematics and Computers in Simulation, 192, 84-110.

The algorithm mimics the foraging behavior of honey badgers, known for their
intelligence, persistence, and fearlessness in hunting prey and raiding beehives.

## Usage

```python
from opt.swarm_intelligence.honey_badger import HoneyBadgerAlgorithm
from opt.benchmark.functions import sphere

optimizer = HoneyBadgerAlgorithm(
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
| `beta` | `float` | `_BETA` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`honey_badger.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/honey_badger.py)
:::
