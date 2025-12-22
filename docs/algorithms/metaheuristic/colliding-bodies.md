# Colliding Bodies Optimization

<span class="badge badge-metaheuristic">Metaheuristic</span>

The implementation of the Colliding Bodies Optimization algorithm.

## Algorithm Overview

The Colliding Bodies Optimization algorithm is inspired by the behavior of colliding
bodies in physics. It aims to find the global minimum of a given objective function.

Example usage:
    optimizer = CollidingBodiesOptimization(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

## Usage

```python
from opt.metaheuristic.colliding_bodies_optimization import CollidingBodiesOptimization
from opt.benchmark.functions import sphere

optimizer = CollidingBodiesOptimization(
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
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |
| `population_size` | `int` | `100` | Number of individuals in population |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`colliding_bodies_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/colliding_bodies_optimization.py)
:::
