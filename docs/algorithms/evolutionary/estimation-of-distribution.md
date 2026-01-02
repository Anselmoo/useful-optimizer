# Estimation Of Distribution Algorithm

<span class="badge badge-evolutionary">Evolutionary</span>

Estimation of Distribution Algorithm optimizer.

## Algorithm Overview

This module implements the Estimation of Distribution Algorithm (EDA) optimizer.
The EDA optimizer is a population-based optimization algorithm that uses a probabilistic model
to estimate the distribution of promising solutions. It iteratively generates new solutions
by sampling from the estimated distribution.

The EstimationOfDistributionAlgorithm class is a subclass of the AbstractOptimizer class
and provides the implementation of the EDA optimizer. It initializes a population, selects
the best individuals based on fitness, estimates the mean and standard deviation of the
selected individuals, and generates new individuals by sampling from the estimated model.
The process is repeated for a specified number of iterations.

## Usage

```python
from opt.evolutionary.estimation_of_distribution_algorithm import EstimationOfDistributionAlgorithm
from opt.benchmark.functions import sphere

optimizer = EstimationOfDistributionAlgorithm(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Population size. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`estimation_of_distribution_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/evolutionary/estimation_of_distribution_algorithm.py)
:::
