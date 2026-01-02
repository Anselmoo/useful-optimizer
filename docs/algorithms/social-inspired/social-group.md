# Social Group Optimizer

<span class="badge badge-social">Social-Inspired</span>

Social Group Optimization Algorithm.

## Algorithm Overview

This module implements the Social Group Optimization (SGO) algorithm,
a social-inspired metaheuristic based on human social behavior.

The algorithm simulates social interaction behaviors including improving,
acquiring knowledge from others, and self-introspection.

## Reference

> Satapathy, S. C., & Naik, A. (2016). Social group optimization (SGO): A new population evolutionary optimization technique. Complex & Intelligent Systems, 2(3), 173-203. DOI: 10.1007/s40747-016-0022-8

[📄 View Paper (DOI: 10.1007/s40747-016-0022-8)](https://doi.org/10.1007/s40747-016-0022-8)

## Usage

```python
from opt.social_inspired.social_group_optimizer import SocialGroupOptimizer
from opt.benchmark.functions import sphere

optimizer = SocialGroupOptimizer(
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
| `population_size` | `int` | `30` | Number of individuals in social group. |
| `max_iter` | `int` | `100` | Maximum iterations. |
| `c` | `float` | `0.2` | Self-introspection coefficient controlling exploration
        intensity. |
| `track_convergence` | `bool` | `False` | Enable convergence history tracking. |
| `early_stopping` | `bool` | `False` | Enable early stopping when improvement
        stagnates. |
| `tolerance` | `float` | `1e-06` | Minimum improvement threshold for early stopping. |
| `patience` | `int` | `10` | Iterations without improvement before early stopping. |
| `verbose` | `bool` | `False` | Print optimization progress. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Social-Inspired Algorithms](/algorithms/social-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`social_group_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/social_inspired/social_group_optimizer.py)
:::
