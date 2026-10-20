# Metaheuristic API

API reference for the 14 classes of `opt.metaheuristic`. This page is generated from the source; see the [algorithm pages](/algorithms/metaheuristic/) for formulations and references.

## Imports

```python
from opt.metaheuristic import (
    ArithmeticOptimizationAlgorithm,
    CollidingBodiesOptimization,
    CrossEntropyMethod,
    EagleStrategy,
    ForensicBasedInvestigationOptimizer,
    HarmonySearch,
    ParticleFilter,
    ShuffledFrogLeapingAlgorithm,
    SineCosineAlgorithm,
    StochasticDiffusionSearch,
    StochasticFractalSearch,
    VariableDepthSearch,
    VariableNeighborhoodSearch,
    VeryLargeScaleNeighborhood,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`ArithmeticOptimizationAlgorithm`](/algorithms/metaheuristic/arithmetic-optimization) | Arithmetic Optimization Algorithm | - |
| [`CollidingBodiesOptimization`](/algorithms/metaheuristic/colliding-bodies-optimization) | Colliding Bodies Optimization | - |
| [`CrossEntropyMethod`](/algorithms/metaheuristic/cross-entropy-method) | Cross-Entropy Method | `elite_frac=0.2`, `noise_decay=0.99` |
| [`EagleStrategy`](/algorithms/metaheuristic/eagle-strategy) | Eagle Strategy | - |
| [`ForensicBasedInvestigationOptimizer`](/algorithms/metaheuristic/forensic-based) | Forensic-Based Investigation Optimizer | - |
| [`HarmonySearch`](/algorithms/metaheuristic/harmony-search) | Harmony Search | `harmony_memory_accepting_rate=0.95`, `pitch_adjusting_rate=0.7`, `bandwidth=0.01` |
| [`ParticleFilter`](/algorithms/metaheuristic/particle-filter) | Sequential Monte Carlo Particle Filter | `inertia=0.7`, `cognitive=1.5`, `social=1.5` |
| [`ShuffledFrogLeapingAlgorithm`](/algorithms/metaheuristic/shuffled-frog-leaping-algorithm) | Shuffled Frog Leaping Algorithm | `cut=2` |
| [`SineCosineAlgorithm`](/algorithms/metaheuristic/sine-cosine-algorithm) | Sine Cosine Algorithm | `r1_cut=0.5`, `r2_cut=0.5` |
| [`StochasticDiffusionSearch`](/algorithms/metaheuristic/stochastic-diffusion-search) | Stochastic Diffusion Search | - |
| [`StochasticFractalSearch`](/algorithms/metaheuristic/stochastic-fractal-search) | Stochastic Fractal Search | `diffusion_parameter=0.5` |
| [`VariableDepthSearch`](/algorithms/metaheuristic/variable-depth-search) | Variable Depth Search | `max_depth=20` |
| [`VariableNeighborhoodSearch`](/algorithms/metaheuristic/variable-neighborhood-search) | Variable Neighborhood Search | `neighborhood_size=10` |
| [`VeryLargeScaleNeighborhood`](/algorithms/metaheuristic/very-large-scale-neighborhood-search) | Very Large Scale Neighborhood Search | `neighborhood_size=10` |
