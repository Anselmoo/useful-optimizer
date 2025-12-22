# Abstract Optimizer

The `AbstractOptimizer` class is the base class that all optimization algorithms in Useful Optimizer inherit from. It defines the common interface and shared functionality.

## Class Definition

```python
from opt.abstract_optimizer import AbstractOptimizer
```

## Constructor

```python
class AbstractOptimizer(ABC):
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 30,
        track_history: bool = False
    ) -> None:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable[[np.ndarray], float]` | Required | The objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of the search space |
| `upper_bound` | `float` | Required | Upper bound of the search space |
| `dim` | `int` | Required | Dimensionality of the problem |
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `population_size` | `int` | `30` | Number of individuals in population |
| `track_history` | `bool` | `False` | Whether to track optimization history |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `func` | `Callable` | The objective function |
| `lower_bound` | `float` | Lower bound of search space |
| `upper_bound` | `float` | Upper bound of search space |
| `dim` | `int` | Problem dimensionality |
| `max_iter` | `int` | Maximum iterations |
| `seed` | `int` | Random seed |
| `population_size` | `int` | Population size |
| `track_history` | `bool` | History tracking flag |
| `history` | `dict` | Optimization history (if tracking enabled) |

## Methods

### search()

```python
@abstractmethod
def search(self) -> tuple[np.ndarray, float]:
    """Perform the optimization search.

    Returns:
        tuple: A tuple containing:
            - best_solution (np.ndarray): The best solution found
            - best_fitness (float): The fitness value of the best solution
    """
```

This is the main method that runs the optimization algorithm. It must be implemented by all subclasses.

## History Tracking

When `track_history=True`, the optimizer records:

```python
history = {
    "best_fitness": [],      # Best fitness at each iteration
    "best_solution": [],     # Best solution at each iteration
    "population_fitness": [],# All fitness values per iteration
    "population": []         # All solutions per iteration
}
```

### Example: Accessing History

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import rosenbrock

optimizer = ParticleSwarm(
    func=rosenbrock,
    lower_bound=-5.0,
    upper_bound=10.0,
    dim=10,
    max_iter=100,
    track_history=True
)

best_solution, best_fitness = optimizer.search()

# Plot convergence curve
import matplotlib.pyplot as plt

plt.plot(optimizer.history["best_fitness"])
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.yscale("log")
plt.title("Convergence Curve")
plt.show()
```

## Creating Custom Optimizers

To create a custom optimizer, inherit from `AbstractOptimizer` and implement the `search()` method:

```python
from opt.abstract_optimizer import AbstractOptimizer
import numpy as np

class RandomSearch(AbstractOptimizer):
    """Simple random search optimizer."""

    def search(self) -> tuple[np.ndarray, float]:
        best_solution = None
        best_fitness = float('inf')

        rng = np.random.default_rng(self.seed)

        for _ in range(self.max_iter):
            # Generate random solution
            candidate = rng.uniform(
                self.lower_bound,
                self.upper_bound,
                self.dim
            )

            fitness = self.func(candidate)

            if fitness < best_fitness:
                best_solution = candidate
                best_fitness = fitness

            # Track history if enabled
            if self.track_history:
                self.history["best_fitness"].append(best_fitness)
                self.history["best_solution"].append(best_solution.copy())

        return best_solution, best_fitness
```

## Constants

The `AbstractOptimizer` uses constants defined in `opt.constants`:

```python
from opt.constants import (
    DEFAULT_MAX_ITERATIONS,  # 1000
    DEFAULT_POPULATION_SIZE, # 30
    DEFAULT_SEED             # 42
)
```
