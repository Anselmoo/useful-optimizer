# Metaheuristic Algorithms

General-purpose optimization algorithms that can be applied to a wide variety of problems.

## Overview

Metaheuristic algorithms are high-level problem-independent algorithmic frameworks that provide guidelines for developing heuristic optimization algorithms. They balance exploration and exploitation to efficiently search large solution spaces.

## Available Algorithms

- [Harmony Search](./harmony-search) - Music-inspired optimization
- [Cross Entropy Method](./cross-entropy) - Adaptive importance sampling
- [Sine Cosine Algorithm](./sine-cosine) - Mathematical function-based search
- [Simulated Annealing](../classical/simulated-annealing) - Thermodynamics-inspired (also in Classical)
- [Tabu Search](./tabu-search) - Memory-based search
- [Variable Neighborhood Search](./variable-neighborhood) - Local search strategy

## Usage Example

```python
from opt.metaheuristic import HarmonySearch
from opt.benchmark.functions import rosenbrock

optimizer = HarmonySearch(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=10,
    dim=10,
    max_iter=100
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [API Reference](/api/metaheuristic) - Complete API documentation
- [Benchmark Results](/benchmarks/results) - Performance comparisons
