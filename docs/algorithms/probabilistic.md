# Probabilistic Algorithms

Probabilistic optimization algorithms use probability distributions and statistical methods to guide the search process.

## Overview

| Property | Value |
|----------|-------|
| **Category** | Statistical Methods |
| **Algorithms** | 5 |
| **Best For** | Expensive objectives, uncertainty |
| **Characteristic** | Probability-based decisions |

## Algorithm List

### Bayesian Optimization

Model-based optimization using Gaussian Process regression.

```python
from opt.probabilistic import BayesianOptimizer

optimizer = BayesianOptimizer(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=50,
    n_initial=10,
)
```

### Sequential Monte Carlo

Particle-based approximation for complex distributions.

```python
from opt.probabilistic import SequentialMonteCarlo

optimizer = SequentialMonteCarlo(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=500,
)
```

### Complete Algorithm List

| Algorithm | Method | Module |
|-----------|--------|--------|
| Adaptive Metropolis | MCMC with adaptive proposals | `adaptive_metropolis` |
| Bayesian Optimizer | Gaussian Process | `bayesian_optimizer` |
| Linear Discriminant Analysis | Statistical classification | `linear_discriminant_analysis` |
| Parzen Tree Estimator | Kernel density estimation | `parzen_tree_stimator` |
| Sequential Monte Carlo | Particle filtering | `sequential_monte_carlo` |

## See Also

- [API Reference: Probabilistic](../api/probabilistic.md)
