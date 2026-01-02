# Marine Predators Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Marine Predators Algorithm (MPA).

## Algorithm Overview

This module implements the Marine Predators Algorithm, a nature-inspired
metaheuristic based on the foraging strategy of ocean predators.

The algorithm mimics the Lévy and Brownian motion strategies used by marine
predators when hunting prey, with the choice of movement depending on the
velocity ratio between predator and prey.

## Reference

> Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020). Marine Predators Algorithm: A nature-inspired metaheuristic. Expert Systems with Applications, 152, 113377. DOI: 10.1016/j.eswa.2020.113377

[📄 View Paper (DOI: 10.1016/j.eswa.2020.113377)](https://doi.org/10.1016/j.eswa.2020.113377)

## Usage

```python
from opt.swarm_intelligence.marine_predators_algorithm import MarinePredatorsOptimizer
from opt.benchmark.functions import sphere

optimizer = MarinePredatorsOptimizer(
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
| `population_size` | `int` | `100` | Number of predators/prey. |
| `fads` | `float` | `_FADs_EFFECT_PROB` | Fish Aggregating Devices effect probability. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`marine_predators_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/marine_predators_algorithm.py)
:::
