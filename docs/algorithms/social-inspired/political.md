# Political Optimizer

<span class="badge badge-social">Social-Inspired</span>

Political Optimizer Algorithm.

## Algorithm Overview

This module implements the Political Optimizer, a social-inspired metaheuristic
algorithm based on political strategies and election processes.

The algorithm simulates political party behavior including constituency
allocation, party switching, and election campaigns.

## Reference

> Askari, Q., Younas, I., & Saeed, M. (2020). Political Optimizer: A novel socio-inspired meta-heuristic for global optimization. Knowledge-Based Systems, 195, 105709. DOI: 10.1016/j.knosys.2020.105709

[📄 View Paper (DOI: 10.1016/j.knosys.2020.105709)](https://doi.org/10.1016/j.knosys.2020.105709)

## Usage

```python
from opt.social_inspired.political_optimizer import PoliticalOptimizer
from opt.benchmark.functions import sphere

optimizer = PoliticalOptimizer(
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
| `population_size` | `int` | `30` | Number of politicians in the election. |
| `max_iter` | `int` | `100` | Maximum iterations (election cycles). |
| `num_parties` | `int` | `5` | Number of political parties (clusters). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Social-Inspired Algorithms](/algorithms/social-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`political_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/social_inspired/political_optimizer.py)
:::
