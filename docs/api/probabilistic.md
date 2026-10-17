# Probabilistic API

API reference for the 5 classes of `opt.probabilistic`. This page is generated from the source; see the [algorithm pages](/algorithms/probabilistic/) for formulations and references.

## Imports

```python
from opt.probabilistic import (
    AdaptiveMetropolisOptimizer,
    BayesianOptimizer,
    LDAnalysis,
    SequentialMonteCarloOptimizer,
    ParzenTreeEstimator,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`AdaptiveMetropolisOptimizer`](/algorithms/probabilistic/adaptive-metropolis) | Adaptive Metropolis Algorithm | `initial_temp=10.0`, `final_temp=0.01`, `adaptation_start=100` |
| [`BayesianOptimizer`](/algorithms/probabilistic/bayesian-optimizer) | Bayesian Optimization | `n_initial=10`, `xi=0.01` |
| [`LDAnalysis`](/algorithms/probabilistic/linear-discriminant-analysis) | Linear Discriminant Analysis Optimizer | `number_of_labels=20`, `unique_classes=2` |
| [`SequentialMonteCarloOptimizer`](/algorithms/probabilistic/sequential-monte-carlo) | Sequential Monte Carlo Optimization | `initial_temp=10.0`, `final_temp=0.1` |
| [`ParzenTreeEstimator`](/algorithms/probabilistic/parzen-tree-estimator) | Tree-structured Parzen Estimator | `gamma=0.15`, `bandwidth=0.2`, `n_samples=None`, `selection_strategy='difference'` |
