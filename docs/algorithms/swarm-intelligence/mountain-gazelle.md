# Mountain Gazelle Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Mountain Gazelle Optimizer.

## Algorithm Overview

Implementation based on:
Abdollahzadeh, B., Gharehchopogh, F.S., Khodadadi, N. & Mirjalili, S. (2022).
Mountain Gazelle Optimizer: A new Nature-inspired Metaheuristic Algorithm
for Global Optimization Problems.
Advances in Engineering Software, 174, 103282.

The algorithm mimics the social and territorial behaviors of mountain gazelles,
including grazing, mating, and avoiding predators.

## Usage

```python
from opt.swarm_intelligence.mountain_gazelle import MountainGazelleOptimizer
from opt.benchmark.functions import sphere

optimizer = MountainGazelleOptimizer(
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
View the implementation: [`mountain_gazelle.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/mountain_gazelle.py)
:::
