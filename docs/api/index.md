# API Reference

This section provides the complete API documentation for Useful Optimizer, auto-generated from Python docstrings.

## Module Structure

```
opt/
├── abstract_optimizer.py     # Base optimizer class
├── benchmark/                # Benchmark functions
│   └── functions.py
├── swarm_intelligence/       # 57 swarm algorithms
├── evolutionary/             # 6 evolutionary algorithms
├── gradient_based/           # 11 gradient-based optimizers
├── classical/                # 9 classical methods
├── metaheuristic/            # 14 metaheuristic algorithms
├── physics_inspired/         # 4 physics-based algorithms
├── social_inspired/          # 4 social behavior algorithms
├── probabilistic/            # 5 probabilistic methods
├── constrained/              # 5 constrained optimizers
└── multi_objective/          # 4 multi-objective algorithms
```

## Quick Links

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } **Base Classes**

    ---

    Core abstract classes and interfaces

    [:octicons-arrow-right-24: Base Classes](base.md)

-   :material-bee:{ .lg .middle } **Swarm Intelligence**

    ---

    57 nature-inspired swarm algorithms

    [:octicons-arrow-right-24: Swarm Intelligence](swarm-intelligence.md)

-   :material-dna:{ .lg .middle } **Evolutionary**

    ---

    6 evolutionary and genetic algorithms

    [:octicons-arrow-right-24: Evolutionary](evolutionary.md)

-   :material-function-variant:{ .lg .middle } **Gradient-Based**

    ---

    11 gradient descent variants

    [:octicons-arrow-right-24: Gradient-Based](gradient-based.md)

-   :material-target:{ .lg .middle } **Classical**

    ---

    9 traditional optimization methods

    [:octicons-arrow-right-24: Classical](classical.md)

-   :material-lightbulb:{ .lg .middle } **Metaheuristic**

    ---

    14 problem-independent frameworks

    [:octicons-arrow-right-24: Metaheuristic](metaheuristic.md)

-   :material-atom:{ .lg .middle } **Physics-Inspired**

    ---

    4 physics-based algorithms

    [:octicons-arrow-right-24: Physics-Inspired](physics-inspired.md)

-   :material-account-group:{ .lg .middle } **Social-Inspired**

    ---

    4 social behavior algorithms

    [:octicons-arrow-right-24: Social-Inspired](social-inspired.md)

-   :material-chart-bell-curve:{ .lg .middle } **Probabilistic**

    ---

    5 probabilistic methods

    [:octicons-arrow-right-24: Probabilistic](probabilistic.md)

-   :material-lock:{ .lg .middle } **Constrained**

    ---

    5 constrained optimization methods

    [:octicons-arrow-right-24: Constrained](constrained.md)

-   :material-arrow-split-vertical:{ .lg .middle } **Multi-Objective**

    ---

    4 multi-objective algorithms

    [:octicons-arrow-right-24: Multi-Objective](multi-objective.md)

-   :material-test-tube:{ .lg .middle } **Benchmark Functions**

    ---

    Test functions for optimization

    [:octicons-arrow-right-24: Benchmark Functions](benchmark-functions.md)

</div>

## Common Interface

All optimizers share a common interface inherited from `AbstractOptimizer`:

```python
from opt.abstract_optimizer import AbstractOptimizer

class AbstractOptimizer:
    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 50,
        track_history: bool = False,
    ) -> None:
        ...

    def search(self) -> tuple[np.ndarray, float]:
        """Perform optimization and return (best_solution, best_fitness)."""
        ...
```

## Return Types

All `search()` methods return a tuple of:

1. `np.ndarray` - The best solution found
2. `float` - The fitness value of the best solution

## History Tracking

When `track_history=True`, the optimizer records:

```python
optimizer.history = {
    'best_fitness': [],      # Best fitness at each iteration
    'best_solution': [],     # Best solution at each iteration
    'population_fitness': [],  # All fitness values (if applicable)
    'population': [],        # All solutions (if applicable)
}
```
