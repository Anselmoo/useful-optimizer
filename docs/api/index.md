# API Reference

This section provides detailed API documentation for all modules in Useful Optimizer.

<!-- BEGIN GENERATED: modules (scripts/generate_docs.py) -->

## Module Structure

```
opt/
├── abstract/                # Base classes (AbstractOptimizer, history)
├── swarm_intelligence/      # 56 algorithms
├── evolutionary/            # 6 algorithms
├── gradient_based/          # 11 algorithms
├── classical/               # 9 algorithms
├── metaheuristic/           # 14 algorithms
├── physics_inspired/        # 4 algorithms
├── probabilistic/           # 5 algorithms
├── social_inspired/         # 4 algorithms
├── constrained/             # 5 algorithms
├── multi_objective/         # 3 algorithms
└── benchmark/               # Benchmark functions
```

## Algorithm Modules

- [Swarm Intelligence](./swarm-intelligence): 56 classes. Nature-inspired algorithms based on collective behavior.
- [Evolutionary](./evolutionary): 6 classes. Algorithms inspired by biological evolution.
- [Gradient-Based](./gradient-based): 11 classes. Optimization using gradient information.
- [Classical](./classical): 9 classes. Traditional mathematical optimization methods.
- [Metaheuristic](./metaheuristic): 14 classes. High-level problem-independent algorithms.
- [Physics-Inspired](./physics-inspired): 4 classes. Algorithms based on physical phenomena.
- [Probabilistic](./probabilistic): 5 classes. Stochastic and Bayesian optimization methods.
- [Social-Inspired](./social-inspired): 4 classes. Algorithms inspired by social behaviors.
- [Constrained](./constrained): 5 classes. Optimization with constraints.
- [Multi-Objective](./multi-objective): 3 classes. Pareto-optimal solutions for multiple objectives.

<!-- END GENERATED: modules -->

## Core Modules

- [Abstract Optimizer](./abstract-optimizer): the base class and history tracking shared by every optimizer.
- [Benchmark Functions](./benchmark-functions): standard test functions such as Sphere, Rosenbrock, Rastrigin and Ackley.

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
