# Evolutionary Algorithms

Optimization algorithms inspired by biological evolution and natural selection.

## Overview

Evolutionary algorithms use mechanisms inspired by biological evolution, such as reproduction, mutation, recombination, and selection. They maintain a population of candidate solutions and evolve them over generations.

## Available Algorithms

- [Genetic Algorithm](./genetic-algorithm) - Classic evolutionary algorithm
- [Differential Evolution](./differential-evolution) - Vector-based evolutionary strategy
- [CMA-ES](./cma-es) - Covariance Matrix Adaptation Evolution Strategy
- [Cultural Algorithm](./cultural) - Dual inheritance system
- [Imperialist Competitive Algorithm](./imperialist) - Socio-politically inspired

## Usage Example

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
    population_size=50
)
best_solution, best_fitness = ga.search()
```

## See Also

- [API Reference](/api/evolutionary) - Complete API documentation
- [Swarm Intelligence](../swarm-intelligence/) - Related population-based methods
