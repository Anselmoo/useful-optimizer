# Evolutionary Algorithms API

API reference for evolutionary algorithms in `opt.evolutionary`.

## Module Overview

```python
from opt.evolutionary import (
    GeneticAlgorithm,
    DifferentialEvolution,
    CMAES,
    CulturalAlgorithm,
    ImperialistCompetitive,
)
```

## Common Interface

All evolutionary algorithms inherit from `AbstractOptimizer`:

```python
class EvolutionaryAlgorithm(AbstractOptimizer):
    def __init__(
        self,
        func: Callable,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 50,
        **kwargs
    ):
        pass

    def search(self) -> tuple[np.ndarray, float]:
        pass
```

## Available Algorithms

- `GeneticAlgorithm` - Classic GA with crossover and mutation
- `DifferentialEvolution` - Vector-based evolution strategy
- `CMAES` - Covariance Matrix Adaptation ES
- `CulturalAlgorithm` - Dual inheritance system
- `ImperialistCompetitive` - Socio-political optimization

## Example Usage

```python
from opt.evolutionary import GeneticAlgorithm, DifferentialEvolution
from opt.benchmark.functions import rastrigin

# Genetic Algorithm
ga = GeneticAlgorithm(
    func=rastrigin,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=100,
    population_size=50,
    crossover_rate=0.8,
    mutation_rate=0.1
)
solution, fitness = ga.search()

# Differential Evolution
de = DifferentialEvolution(
    func=rastrigin,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=100,
    population_size=50,
    F=0.8,  # Differential weight
    CR=0.9  # Crossover probability
)
solution, fitness = de.search()
```

## See Also

- [Evolutionary Algorithms](/algorithms/evolutionary/) - Algorithm details
