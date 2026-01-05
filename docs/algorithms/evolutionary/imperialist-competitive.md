# Imperialist Competitive Algorithm

<span class="badge badge-evolutionary">Evolutionary</span>

Imperialist Competitive Algorithm optimizer.

## Algorithm Overview

This module implements the Imperialist Competitive Algorithm (ICA) for solving
optimization problems. The ICA is a population-based algorithm that simulates the
competition between empires and colonies. It starts with a random population and
iteratively improves the solutions by assimilation, revolution, position exchange,
and imperialistic competition.

## Usage

```python
from opt.evolutionary.imperialist_competitive_algorithm import ImperialistCompetitiveAlgorithm
from opt.benchmark.functions import sphere

optimizer = ImperialistCompetitiveAlgorithm(
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
| `dim` | `int` | Required | Problem dimensionality. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `num_empires` | `int` | `15` | Number of initial empires. |
| `population_size` | `int` | `100` | Total number of countries. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `revolution_rate` | `float` | `0.3` | Revolution probability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`imperialist_competitive_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/evolutionary/imperialist_competitive_algorithm.py)
:::
