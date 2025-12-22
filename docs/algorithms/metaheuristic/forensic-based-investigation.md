# Forensic Based Investigation Optimizer

<span class="badge badge-metaheuristic">Metaheuristic</span>

Forensic-Based Investigation Optimization.

## Algorithm Overview

Implementation based on:
Chou, J.S. & Nguyen, N.M. (2020).
FBI inspired meta-optimization.
Applied Soft Computing, 93, 106339.

The algorithm mimics the investigation process used by forensic
investigators, including evidence analysis and suspect tracking.

## Usage

```python
from opt.metaheuristic.forensic_based import ForensicBasedInvestigationOptimizer
from opt.benchmark.functions import sphere

optimizer = ForensicBasedInvestigationOptimizer(
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
| `max_iter` | `int` | Required | Maximum number of iterations |
| `population_size` | `int` | `30` | Number of individuals in population |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`forensic_based.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/forensic_based.py)
:::
