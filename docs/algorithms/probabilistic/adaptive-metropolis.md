# Adaptive Metropolis Optimizer

<span class="badge badge-probabilistic">Probabilistic</span>

Simulated Annealing with Adaptive Metropolis.

## Algorithm Overview

This module implements Simulated Annealing enhanced with Adaptive Metropolis
proposal distribution, a probabilistic optimization method.

The algorithm adapts the proposal covariance based on the history of
accepted samples, improving exploration efficiency.

## Reference

> Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. Bernoulli, 7(2), 223-242. DOI: 10.2307/3318737

[📄 View Paper (DOI: 10.2307/3318737)](https://doi.org/10.2307/3318737)

## Usage

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
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `initial_temp` | `float` | `10.0` | Algorithm-specific parameter |
| `final_temp` | `float` | `0.01` | Algorithm-specific parameter |
| `adaptation_start` | `int` | `100` | Algorithm-specific parameter |

## See Also

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adaptive_metropolis.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/probabilistic/adaptive_metropolis.py)
:::
