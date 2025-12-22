# Probabilistic Optimization Algorithms

Algorithms that use probabilistic models to guide the search.

## Overview

Probabilistic optimization algorithms build and update probability distributions over the solution space, using statistical methods to explore and exploit promising regions.

## Available Algorithms

- [Estimation of Distribution Algorithm (EDA)](./eda) - Builds probability distribution of good solutions
- [Cross Entropy Method](../metaheuristic/cross-entropy) - Adaptive importance sampling
- [Bayesian Optimization](./bayesian) - Gaussian process-based optimization
- [Simulated Annealing](../classical/simulated-annealing) - Probabilistic acceptance criterion

## Usage Example

```python
from opt.probabilistic import EstimationOfDistribution
from opt.benchmark.functions import rastrigin

optimizer = EstimationOfDistribution(
    func=rastrigin,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=100
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [API Reference](/api/probabilistic) - Complete API documentation
