# Sine Cosine Algorithm

<span class="badge badge-metaheuristic">Metaheuristic</span>

Sine Cosine Algorithm optimization algorithm.

## Algorithm Overview

This module implements the Sine Cosine Algorithm (SCA) optimization algorithm.
SCA is a population-based metaheuristic algorithm inspired by the sine and cosine
functions. It is commonly used for solving optimization problems.

The SineCosineAlgorithm class provides an implementation of the SCA algorithm. It takes
an objective function, lower and upper bounds of the search space, dimensionality of
the search space, and other optional parameters as input. The search method performs
the optimization and returns the best solution found along with its fitness value.

## Usage

```python
from opt.metaheuristic.sine_cosine_algorithm import SineCosineAlgorithm
from opt.benchmark.functions import sphere

optimizer = SineCosineAlgorithm(
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
| `func` | `Callable` | Required | The objective function to be optimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `population_size` | `int` | `100` | The size of the population (default: 100). |
| `max_iter` | `int` | `1000` | The maximum number of iterations (default: 1000). |
| `r1_cut` | `float` | `0.5` | The threshold for selecting the sine update rule (default: 0. |
| `r2_cut` | `float` | `0.5` | The threshold for selecting the cosine update rule (default: 0. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation (default: None). |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`sine_cosine_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/sine_cosine_algorithm.py)
:::
