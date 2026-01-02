# Grey Wolf Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Grey Wolf Optimizer (GWO) Algorithm.

## Algorithm Overview

!!! warning

    This module is still under development and is not yet ready for use.

This module implements the Grey Wolf Optimizer (GWO) algorithm. GWO is a metaheuristic
optimization algorithm inspired by grey wolves. The algorithm mimics the leadership
hierarchy and hunting mechanism of grey wolves in nature. Four types of grey wolves
such as alpha, beta, delta, and omega are employed for simulating the hunting behavior.

The GWO algorithm is used to solve optimization problems by iteratively trying to
improve a candidate solution with regard to a given measure of quality, or fitness
function.

## Usage

```python
from opt.swarm_intelligence.grey_wolf_optimizer import GreyWolfOptimizer
from opt.benchmark.functions import sphere

optimizer = GreyWolfOptimizer(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Pack size (number of wolves). |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`grey_wolf_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/grey_wolf_optimizer.py)
:::
