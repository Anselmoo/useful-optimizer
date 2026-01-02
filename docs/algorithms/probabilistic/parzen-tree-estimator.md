# Parzen Tree Estimator

<span class="badge badge-probabilistic">Probabilistic</span>

Parzen Tree Estimator optimizer.

## Algorithm Overview

The Parzen Tree Estimator optimizer is an algorithm that uses the Parzen Tree Estimator
technique to search for the optimal solution of a given function within a specified
search space. It is particularly useful for optimization problems where the objective
function is expensive to evaluate.

The Parzen Tree Estimator algorithm works by maintaining a population of
hyperparameters and their corresponding scores. It segments the population into two
distributions based on the scores and fits Gaussian kernel density estimators to each
distribution. It then samples hyperparameters from the low score distribution and
selects the hyperparameters with the highest score difference or ratio between the
low and high score distributions. This process is iteratively repeated to search
for the optimal solution.

This implementation of the Parzen Tree Estimator optimizer provides a flexible and
customizable framework for solving optimization problems. It allows users to specify
the objective function, search space, population size, maximum number of iterations,
selection strategy, and other parameters.

## Usage

```python
from opt.probabilistic.parzen_tree_stimator import ParzenTreeEstimator
from opt.benchmark.functions import sphere

optimizer = ParzenTreeEstimator(
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
| `population_size` | `int` | `100` | Number of observations to maintain for KDE fitting. |
| `max_iter` | `int` | `1000` | Maximum TPE iterations. |
| `gamma` | `float` | `0.15` | Quantile for splitting observations into good/bad. |
| `bandwidth` | `float` | `0.2` | Gaussian kernel bandwidth for KDE. |
| `n_samples` | `int  \|  None` | `None` | Number of candidates to sample from good KDE. |
| `selection_strategy` | `str` | `'difference'` | Strategy for selecting next point:
        "difference" or "ratio". |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`parzen_tree_stimator.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/probabilistic/parzen_tree_stimator.py)
:::
