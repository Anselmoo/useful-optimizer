# Evolutionary API

API reference for the 6 classes of `opt.evolutionary`. This page is generated from the source; see the [algorithm pages](/algorithms/evolutionary/) for formulations and references.

## Imports

```python
from opt.evolutionary import (
    CMAESAlgorithm,
    CulturalAlgorithm,
    DifferentialEvolution,
    EstimationOfDistributionAlgorithm,
    GeneticAlgorithm,
    ImperialistCompetitiveAlgorithm,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`CMAESAlgorithm`](/algorithms/evolutionary/cma-es) | Covariance Matrix Adaptation Evolution Strategy | `sigma_init=0.5`, `epsilon=1e-09` |
| [`CulturalAlgorithm`](/algorithms/evolutionary/cultural-algorithm) | Cultural Algorithm | `belief_space_size=20`, `scaling_factor=0.5`, `mutation_probability=0.5`, `elitism=0.1` |
| [`DifferentialEvolution`](/algorithms/evolutionary/differential-evolution) | Differential Evolution | `F=0.5`, `CR=0.7` |
| [`EstimationOfDistributionAlgorithm`](/algorithms/evolutionary/estimation-of-distribution-algorithm) | Estimation of Distribution Algorithm | - |
| [`GeneticAlgorithm`](/algorithms/evolutionary/genetic-algorithm) | Genetic Algorithm | `tournament_size=3`, `crossover_rate=0.7` |
| [`ImperialistCompetitiveAlgorithm`](/algorithms/evolutionary/imperialist-competitive-algorithm) | Imperialist Competitive Algorithm | `num_empires=15`, `revolution_rate=0.3` |
