# Multi-Objective API

API reference for the 3 classes of `opt.multi_objective`. This page is generated from the source; see the [algorithm pages](/algorithms/multi-objective/) for formulations and references.

## Imports

```python
from opt.multi_objective import (
    MOEAD,
    NSGAII,
    SPEA2,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`MOEAD`](/algorithms/multi-objective/moead) | Multi-Objective Evolutionary Algorithm based on Decomposition | `objectives`, `n_neighbors=20` |
| [`NSGAII`](/algorithms/multi-objective/nsga-ii) | Non-dominated Sorting Genetic Algorithm II | `objectives`, `crossover_prob=_CROSSOVER_PROBABILITY`, `mutation_prob=None`, `tournament_size=_TOURNAMENT_SIZE`, `eta_c=_SBX_DISTRIBUTION_INDEX`, `eta_m=_POLYNOMIAL_MUTATION_INDEX` |
| [`SPEA2`](/algorithms/multi-objective/spea2) | Strength Pareto Evolutionary Algorithm 2 | `objectives`, `archive_size=100` |
