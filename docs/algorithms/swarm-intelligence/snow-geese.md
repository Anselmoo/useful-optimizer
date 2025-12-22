# Snow Geese Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Snow Geese Optimization Algorithm (SGOA).

## Algorithm Overview

This module implements the Snow Geese Optimization Algorithm, a swarm
intelligence algorithm inspired by the migration behavior of snow geese.

Snow geese migrate in large flocks following V-formation patterns,
with leaders guiding the flock and rotation of positions for energy efficiency.

## Reference

> Jiang, H., Yang, Y., Ping, W., & Dong, Y. (2023). A novel hybrid algorithm based on Snow Geese and Differential Evolution for global optimization. Applied Soft Computing, 139, 110235. DOI: 10.1016/j.asoc.2023.110235

[📄 View Paper (DOI: 10.1016/j.asoc.2023.110235)](https://doi.org/10.1016/j.asoc.2023.110235)

## Usage

```python
from opt.swarm_intelligence.snow_geese import SnowGeeseOptimizer
from opt.benchmark.functions import sphere

optimizer = SnowGeeseOptimizer(
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
View the implementation: [`snow_geese.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/snow_geese.py)
:::
