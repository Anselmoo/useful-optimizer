# Cultural Algorithm

<span class="badge badge-evolutionary">Evolutionary</span>

Cultural Algorithm implementation.

## Algorithm Overview

This module provides an implementation of the Cultural Algorithm optimizer. The
Cultural Algorithm is a population-based optimization algorithm that combines
individual learning (exploitation) with social learning (exploration) to search
for the best solution to a given optimization problem.

The CulturalAlgorithm class is the main class of this module. It inherits from the
AbstractOptimizer class and implements the search method to perform the Cultural
Algorithm search.

Example usage:
    optimizer = CulturalAlgorithm(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.evolutionary.cultural_algorithm import CulturalAlgorithm
from opt.benchmark.functions import sphere

optimizer = CulturalAlgorithm(
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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `population_size` | `int` | `100` | The size of the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `belief_space_size` | `int` | `20` | The size of the belief space. |
| `scaling_factor` | `float` | `0.5` | The scaling factor used in mutation. |
| `mutation_probability` | `float` | `0.5` | The probability of mutation. |
| `elitism` | `float` | `0.1` | The elitism factor. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |

## See Also

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cultural_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/evolutionary/cultural_algorithm.py)
:::
