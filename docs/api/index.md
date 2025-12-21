---
title: API Reference
description: Complete API documentation for Useful Optimizer
---

# API Reference

This page provides a complete reference for the Useful Optimizer API.

---

## Base Classes

### AbstractOptimizer

All optimizers inherit from `AbstractOptimizer`, providing a consistent interface.

```python
from opt.abstract_optimizer import AbstractOptimizer
```

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | `Callable[[np.ndarray], float]` | Objective function to minimize |
| `lower_bound` | `float` | Lower boundary of search space |
| `upper_bound` | `float` | Upper boundary of search space |
| `dim` | `int` | Problem dimensionality |
| `max_iter` | `int` | Maximum iterations (default varies) |

#### Methods

##### `search() -> tuple[np.ndarray, float]`

Execute the optimization algorithm.

**Returns:**

- `best_solution`: NumPy array of shape `(dim,)` containing the best found solution
- `best_fitness`: Float value of the objective function at the best solution

---

## Import Patterns

### Categorical Imports (Recommended)

```python
# Swarm Intelligence
from opt.swarm_intelligence import (
    ParticleSwarm,
    AntColony,
    BatAlgorithm,
    BeeAlgorithm,
    CatSwarmOptimization,
    CuckooSearch,
    FireflyAlgorithm,
    ArtificialFishSwarmAlgorithm,
    GlowwormSwarmOptimization,
    GreyWolfOptimizer,
    SquirrelSearch,
    WhaleOptimizationAlgorithm,
)

# Evolutionary
from opt.evolutionary import (
    CMAEvolutionStrategy,
    CulturalAlgorithm,
    DifferentialEvolution,
    EstimationOfDistributionAlgorithm,
    GeneticAlgorithm,
    ImperialistCompetitiveAlgorithm,
)

# Gradient-Based
from opt.gradient_based import (
    Adadelta,
    Adagrad,
    AdaptiveMomentEstimation,
    AdaMax,
    AdamW,
    AMSGrad,
    Nadam,
    NesterovAcceleratedGradient,
    RMSprop,
    SGDMomentum,
    StochasticGradientDescent,
)

# Classical
from opt.classical import (
    BFGS,
    ConjugateGradient,
    HillClimbing,
    LBFGS,
    NelderMead,
    Powell,
    SimulatedAnnealing,
    TabuSearch,
    TrustRegion,
)

# Metaheuristic
from opt.metaheuristic import (
    CollidingBodiesOptimization,
    CrossEntropyMethod,
    EagleStrategy,
    HarmonySearch,
    ParticleFilter,
    ShuffledFrogLeapingAlgorithm,
    SineCosineAlgorithm,
    StochasticDiffusionSearch,
    StochasticFractalSearch,
    VariableDepthSearch,
    VariableNeighbourhoodSearch,
    VeryLargeScaleNeighborhoodSearch,
)

# Constrained
from opt.constrained import (
    AugmentedLagrangianMethod,
    SuccessiveLinearProgramming,
)

# Probabilistic
from opt.probabilistic import (
    LinearDiscriminantAnalysis,
    ParzenTreeEstimator,
)
```

### Direct Imports

```python
from opt import ParticleSwarm, AdamW, BFGS
```

---

## Benchmark Functions

```python
from opt.benchmark.functions import (
    sphere,
    rosenbrock,
    ackley,
    shifted_ackley,
    rastrigin,
    # ... and more
)
```

### Function Signatures

All benchmark functions follow the same pattern:

```python
def benchmark_function(x: np.ndarray) -> float:
    """
    Args:
        x: Input vector of shape (n,)

    Returns:
        Scalar objective value
    """
    ...
```

---

## Demo Utility

```python
from opt.demo import run_demo

# Run default demo for any optimizer
run_demo(OptimizerClass)

# Customize demo parameters
run_demo(
    OptimizerClass,
    max_iter=200,
    population_size=50,
    **kwargs
)
```

---

## Type Hints

Useful Optimizer provides type hints for better IDE support:

```python
from typing import Callable
import numpy as np

def my_function(x: np.ndarray) -> float:
    return float(np.sum(x**2))

# Type-safe usage
optimizer: AbstractOptimizer = ParticleSwarm(
    func=my_function,
    lower_bound=-5.0,
    upper_bound=5.0,
    dim=10,
)
solution: np.ndarray
fitness: float
solution, fitness = optimizer.search()
```
