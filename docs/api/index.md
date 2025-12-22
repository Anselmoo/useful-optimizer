# API Reference

This section provides detailed API documentation for all modules in Useful Optimizer.

## Module Structure

```
opt/
├── abstract_optimizer.py    # Base class for all optimizers
├── swarm_intelligence/      # 57+ swarm-based algorithms
├── evolutionary/            # 6 evolutionary algorithms
├── gradient_based/          # 11 gradient-based optimizers
├── classical/               # 9 classical methods
├── metaheuristic/           # 12 metaheuristic algorithms
├── constrained/             # 2 constrained optimization
├── probabilistic/           # 2 probabilistic methods
└── benchmark/               # Benchmark functions
```

## Core Modules

### [Abstract Optimizer](./abstract-optimizer)

The base class that all optimizers inherit from. Defines the common interface and shared functionality.

### [Benchmark Functions](./benchmark-functions)

Standard test functions for evaluating optimizer performance, including:
- Sphere function
- Rosenbrock function
- Rastrigin function
- Ackley function
- Griewank function

## Algorithm Modules

### [Swarm Intelligence](./swarm-intelligence)

Nature-inspired population-based algorithms:
- Particle Swarm Optimization
- Ant Colony Optimization
- Firefly Algorithm
- Grey Wolf Optimizer
- Whale Optimization Algorithm
- And 50+ more...

### [Evolutionary](./evolutionary)

Evolution-based optimization methods:
- Genetic Algorithm
- Differential Evolution
- CMA-ES
- Cultural Algorithm
- Estimation of Distribution Algorithm

### [Gradient-Based](./gradient-based)

Gradient descent variants and adaptive methods:
- SGD with Momentum
- Adam
- AdamW
- RMSprop
- Adagrad
- Adadelta
- AMSGrad
- Nadam

### [Classical](./classical)

Traditional mathematical optimization:
- BFGS
- L-BFGS
- Nelder-Mead
- Powell's Method
- Simulated Annealing
- Conjugate Gradient

### [Metaheuristic](./metaheuristic)

High-level optimization frameworks:
- Harmony Search
- Cross Entropy Method
- Sine Cosine Algorithm
- Variable Neighbourhood Search

### [Constrained](./constrained)

Methods for constrained optimization:
- Augmented Lagrangian Method
- Successive Linear Programming

### [Probabilistic](./probabilistic)

Probability-based optimization:
- Parzen Tree Estimator (TPE)
- Linear Discriminant Analysis

## Type Hints

All public APIs use Python type hints for better IDE support:

```python
from typing import Callable
import numpy as np

def search(self) -> tuple[np.ndarray, float]:
    """Run the optimization search.

    Returns:
        tuple: A tuple containing:
            - best_solution (np.ndarray): The best solution found
            - best_fitness (float): The fitness value of the best solution
    """
    ...
```

## Docstring Convention

All modules follow Google-style docstrings:

```python
def __init__(
    self,
    func: Callable[[np.ndarray], float],
    lower_bound: float,
    upper_bound: float,
    dim: int,
    max_iter: int = 100,
    population_size: int = 30
) -> None:
    """Initialize the optimizer.

    Args:
        func: Objective function to minimize. Takes a numpy array
            and returns a scalar fitness value.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of individuals in the population.

    Raises:
        ValueError: If dim < 1 or max_iter < 1.
    """
```
