# Stochastic Fractal Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Stochastic Diffusion Search optimizer.

## Algorithm Overview

This module implements the Stochastic Fractal Search optimizer, which is an
optimization algorithm used to find the minimum of a given function.

The Stochastic Fractal Search algorithm works by maintaining a population of
individuals and iteratively updating them based on their scores. At each iteration,
a best individual is selected, and other individuals in the population undergo a
diffusion phase to explore the search space. The algorithm continues for a specified
number of iterations or until a termination condition is met.

## Usage

```python
from opt.metaheuristic.stochastic_fractal_search import StochasticFractalSearch
from opt.benchmark.functions import sphere

optimizer = StochasticFractalSearch(
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
| `diffusion_parameter` | `float` | `0.5` | The diffusion parameter. |
| `seed` | `int  \|  None` | `None` | The seed for the random number generator. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`stochastic_fractal_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/stochastic_fractal_search.py)
:::
