# African Buffalo Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

African Buffalo Optimization Algorithm.

## Algorithm Overview

Implementation based on:
Odili, J.B., Kahar, M.N.M. & Anwar, S. (2015).
African Buffalo Optimization: A Swarm-Intelligence Technique.
Procedia Computer Science, 76, 443-448.

The algorithm mimics the migratory and herding behavior of African buffalos,
using two key equations: the buffalo's movement toward the best location and
its tendency to explore new areas.

## Usage

```python
from opt.swarm_intelligence.african_buffalo_optimization import AfricanBuffaloOptimizer
from opt.benchmark.functions import sphere

optimizer = AfricanBuffaloOptimizer(
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
| `lp1` | `float` | `_LP1` | Learning parameter 1 controlling exploitation strength. |
| `lp2` | `float` | `_LP2` | Learning parameter 2 controlling exploration strength. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`african_buffalo_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/african_buffalo_optimization.py)
:::
