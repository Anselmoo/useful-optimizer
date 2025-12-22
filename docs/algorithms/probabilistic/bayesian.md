# Bayesian Optimizer

<span class="badge badge-probabilistic">Probabilistic</span>

Bayesian Optimization.

## Algorithm Overview

This module implements Bayesian Optimization, a probabilistic optimization
technique using Gaussian Process surrogate models.

The algorithm builds a probabilistic model of the objective function and
uses it to select promising points to evaluate.

## Reference

> Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. Advances in Neural Information Processing Systems 25 (NIPS 2012).

## Usage

```python
from opt.probabilistic.bayesian_optimizer import BayesianOptimizer
from opt.benchmark.functions import sphere

optimizer = BayesianOptimizer(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
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
| `n_initial` | `int` | `10` | Algorithm-specific parameter |
| `max_iter` | `int` | `50` | Maximum number of iterations |
| `xi` | `float` | `0.01` | Algorithm-specific parameter |

## See Also

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`bayesian_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/probabilistic/bayesian_optimizer.py)
:::
