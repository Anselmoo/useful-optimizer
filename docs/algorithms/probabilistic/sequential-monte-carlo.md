# Sequential Monte Carlo Optimizer

<span class="badge badge-probabilistic">Probabilistic</span>

Sequential Monte Carlo Optimizer.

## Algorithm Overview

This module implements Sequential Monte Carlo (SMC) optimization,
a probabilistic method using importance sampling and particle resampling.

The algorithm maintains a population of weighted particles that
progressively focus on promising regions of the search space.

## Reference

> Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo Samplers. Journal of the Royal Statistical Society: Series B, 68(3), 411-436. DOI: 10.1111/j.1467-9868.2006.00553.x

[📄 View Paper (DOI: 10.1111/j.1467-9868.2006.00553.x)](https://doi.org/10.1111/j.1467-9868.2006.00553.x)

## Usage

```python
from opt.probabilistic.sequential_monte_carlo import SequentialMonteCarloOptimizer
from opt.benchmark.functions import sphere

optimizer = SequentialMonteCarloOptimizer(
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
| `population_size` | `int` | `50` | Number of particles in SMC population. |
| `max_iter` | `int` | `100` | Maximum SMC iterations. |
| `initial_temp` | `float` | `10.0` | Starting temperature for importance weighting. |
| `final_temp` | `float` | `0.1` | Final temperature for importance weighting. |

## See Also

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`sequential_monte_carlo.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/probabilistic/sequential_monte_carlo.py)
:::
