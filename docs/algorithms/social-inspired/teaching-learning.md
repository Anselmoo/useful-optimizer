# Teaching Learning Optimizer

<span class="badge badge-social">Social-Inspired</span>

Teaching-Learning Based Optimization (TLBO).

## Algorithm Overview

This module implements Teaching-Learning Based Optimization,
a metaheuristic algorithm inspired by the teaching-learning
process in a classroom.

## Reference

> Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011). Teaching-learning-based optimization: A novel method for constrained mechanical design optimization problems. Computer-Aided Design, 43(3), 303-315.

## Usage

```python
from opt.social_inspired.teaching_learning import TeachingLearningOptimizer
from opt.benchmark.functions import sphere

optimizer = TeachingLearningOptimizer(
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
| `population_size` | `int` | `50` | Number of individuals in population |
| `max_iter` | `int` | `500` | Maximum number of iterations |

## See Also

- [Social-Inspired Algorithms](/algorithms/social-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`teaching_learning.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/social_inspired/teaching_learning.py)
:::
