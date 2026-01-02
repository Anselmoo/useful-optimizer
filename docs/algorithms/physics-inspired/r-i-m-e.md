# R I M E Optimizer

<span class="badge badge-physics">Physics-Inspired</span>

RIME Optimization Algorithm.

## Algorithm Overview

This module implements the RIME optimization algorithm, a physics-based
metaheuristic inspired by the natural phenomenon of rime-ice formation.

Rime is a type of ice formed when supercooled water droplets freeze on
contact with a surface. The algorithm simulates this physical process
for optimization.

## Reference

> Su, H., Zhao, D., Heidari, A. A., Liu, L., Zhang, X., Mafarja, M., & Chen, H. (2023). RIME: A physics-based optimization. Neurocomputing, 532, 183-214. DOI: 10.1016/j.neucom.2023.02.010

[📄 View Paper (DOI: 10.1016/j.neucom.2023.02.010)](https://doi.org/10.1016/j.neucom.2023.02.010)

## Usage

```python
from opt.physics_inspired.rime_optimizer import RIMEOptimizer
from opt.benchmark.functions import sphere

optimizer = RIMEOptimizer(
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
| `population_size` | `int` | `30` | Population size (number of agents). |
| `max_iter` | `int` | `100` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Physics-Inspired Algorithms](/algorithms/physics-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`rime_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/physics_inspired/rime_optimizer.py)
:::
