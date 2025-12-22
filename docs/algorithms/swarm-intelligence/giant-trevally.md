# Giant Trevally Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Giant Trevally Optimizer (GTO).

## Algorithm Overview

This module implements the Giant Trevally Optimizer, a bio-inspired
metaheuristic algorithm based on the hunting behavior of giant trevally fish.

Giant trevallies are apex predators known for their remarkable hunting
strategy of jumping out of water to catch birds and cooperative hunting.

## Reference

> Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640. DOI: 10.1109/ACCESS.2022.3223388

[📄 View Paper (DOI: 10.1109/ACCESS.2022.3223388)](https://doi.org/10.1109/ACCESS.2022.3223388)

## Usage

```python
from opt.swarm_intelligence.giant_trevally import GiantTrevallyOptimizer
from opt.benchmark.functions import sphere

optimizer = GiantTrevallyOptimizer(
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
| `population_size` | `int` | `30` | Number of individuals in population |
| `max_iter` | `int` | `100` | Maximum number of iterations |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`giant_trevally.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/giant_trevally.py)
:::
