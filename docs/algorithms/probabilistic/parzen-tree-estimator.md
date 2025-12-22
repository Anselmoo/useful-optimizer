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
| `func` | `Callable` | Required | The objective function to be optimized. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `lower_bound` | `float` | Required | The lower bound(s) of the search space. |
| `upper_bound` | `float` | Required | The upper bound(s) of the search space. |
| `population_size` | `int` | `100` | The size of the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `gamma` | `float` | `0.15` | The quantile value used to segment the distributions. |
| `bandwidth` | `float` | `0.2` | The bandwidth of the Gaussian kernel used in the Parzen Tree Estimator. |
| `n_samples` | `int  \|  None` | `None` | The number of samples to draw from the estimated distributions. |
| `selection_strategy` | `str` | `'difference'` | The selection strategy used to choose the next set of hyperparameters. |
| `seed` | `int  \|  None` | `None` | The seed value for the random number generator. |

## See Also

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`parzen_tree_stimator.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/probabilistic/parzen_tree_stimator.py)
:::
