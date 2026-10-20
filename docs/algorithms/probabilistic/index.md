# 📊 Probabilistic Algorithms

Stochastic and Bayesian optimization methods.

This category contains **5 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Adaptive Metropolis Algorithm](./adaptive-metropolis) | AM | 2001 | Adaptive Metropolis (AM) algorithm with covariance adaptation. |
| [Bayesian Optimization](./bayesian-optimizer) | BO | 2012 | Bayesian Optimization (BO) using Gaussian Process surrogates. |
| [Linear Discriminant Analysis Optimizer](./linear-discriminant-analysis) | LDA-OPT | 1936 | LDA-guided Optimization using Linear Discriminant Analysis. |
| [Sequential Monte Carlo Optimization](./sequential-monte-carlo) | SMC | 2006 | Sequential Monte Carlo (SMC) optimization with particle filtering. |
| [Tree-structured Parzen Estimator](./parzen-tree-estimator) | TPE | 2011 | Tree-structured Parzen Estimator (TPE) for hyperparameter optimization. |

## Usage

Every optimizer shares the same interface:

```python
from opt.probabilistic.adaptive_metropolis import AdaptiveMetropolisOptimizer
from opt.benchmark.functions import sphere

optimizer = AdaptiveMetropolisOptimizer(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)
