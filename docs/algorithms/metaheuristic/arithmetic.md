# Arithmetic Optimization Algorithm

<span class="badge badge-metaheuristic">Metaheuristic</span>

Arithmetic Optimization Algorithm (AOA) implementation.

## Algorithm Overview

This module implements the Arithmetic Optimization Algorithm, a math-inspired
metaheuristic optimization algorithm based on arithmetic operators.

## Reference

> Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M., & Gandomi, A. H. (2021). The arithmetic optimization algorithm. Computer Methods in Applied Mechanics and Engineering, 376, 113609.

## Usage

```python
from opt.metaheuristic.arithmetic_optimization import ArithmeticOptimizationAlgorithm
from opt.benchmark.functions import sphere

optimizer = ArithmeticOptimizationAlgorithm(
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
| `population_size` | `int` | `30` | Number of candidate solutions. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`arithmetic_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/arithmetic_optimization.py)
:::
