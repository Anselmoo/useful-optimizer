# SPEA2

<span class="badge badge-multi">Multi-Objective</span>

SPEA2 (Strength Pareto Evolutionary Algorithm 2) implementation.

## Algorithm Overview

This module implements SPEA2, an improved version of the Strength Pareto
Evolutionary Algorithm for multi-objective optimization.

## Reference

> Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the strength pareto evolutionary algorithm. TIK-Report 103, ETH Zurich.

## Usage

```python
from opt.multi_objective.spea2 import SPEA2
from opt.benchmark.functions import sphere

optimizer = SPEA2(
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
| `objectives` | `list[Callable]` | Required | Algorithm-specific parameter |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `max_iter` | `int` | Required | Maximum number of iterations |
| `population_size` | `int` | `100` | Number of individuals in population |
| `archive_size` | `int` | `100` | Algorithm-specific parameter |

## See Also

- [Multi-Objective Algorithms](/algorithms/multi-objective/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`spea2.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/multi_objective/spea2.py)
:::
