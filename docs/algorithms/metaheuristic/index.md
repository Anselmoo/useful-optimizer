# 🔧 Metaheuristic Algorithms

High-level problem-independent algorithms.

This category contains **14 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Arithmetic Optimization Algorithm](./arithmetic-optimization) | AOA | 2021 | Arithmetic Optimization Algorithm (AOA) optimization algorithm. |
| [Colliding Bodies Optimization](./colliding-bodies-optimization) | CBO | 2014 | Colliding Bodies Optimization (CBO) optimization algorithm. |
| [Cross-Entropy Method](./cross-entropy-method) | CEM | 1999 | Cross-Entropy Method (CEM) optimization algorithm. |
| [Eagle Strategy](./eagle-strategy) | ES | 2010 | Eagle Strategy (ES) optimization algorithm. |
| [Forensic-Based Investigation Optimizer](./forensic-based) | FBI | 2020 | Forensic-Based Investigation Optimizer (FBI) optimization algorithm. |
| [Harmony Search](./harmony-search) | HS | 2001 | Harmony Search (HS) optimization algorithm. |
| [Sequential Monte Carlo Particle Filter](./particle-filter) | SMC-PF | 1993 | Sequential Monte Carlo Particle Filter (SMC-PF) optimization algorithm. |
| [Shuffled Frog Leaping Algorithm](./shuffled-frog-leaping-algorithm) | SFLA | 2006 | Shuffled Frog Leaping Algorithm (SFLA) optimization algorithm. |
| [Sine Cosine Algorithm](./sine-cosine-algorithm) | SCA | 2016 | Sine Cosine Algorithm (SCA) optimization algorithm. |
| [Stochastic Diffusion Search](./stochastic-diffusion-search) | SDS | 1989 | Stochastic Diffusion Search (SDS) optimization algorithm. |
| [Stochastic Fractal Search](./stochastic-fractal-search) | SFS | 2015 | Stochastic Fractal Search (SFS) optimization algorithm. |
| [Variable Depth Search](./variable-depth-search) | VDS | 1973 | Variable Depth Search (VDS) optimization algorithm. |
| [Variable Neighborhood Search](./variable-neighborhood-search) | VNS | 1997 | Variable Neighborhood Search (VNS) optimization algorithm. |
| [Very Large Scale Neighborhood Search](./very-large-scale-neighborhood-search) | VLSN | 2000 | Very Large Scale Neighborhood Search (VLSN) optimization algorithm. |

## Usage

Every optimizer shares the same interface:

```python
from opt.metaheuristic.arithmetic_optimization import ArithmeticOptimizationAlgorithm
from opt.benchmark.functions import sphere

optimizer = ArithmeticOptimizationAlgorithm(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)
