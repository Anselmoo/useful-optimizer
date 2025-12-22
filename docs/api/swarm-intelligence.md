# Swarm Intelligence API

API reference for swarm intelligence algorithms in `opt.swarm_intelligence`.

## Module Overview

```python
from opt.swarm_intelligence import (
    ParticleSwarm,
    AntColony,
    ArtificialBeeColony,
    FireflyAlgorithm,
    BatAlgorithm,
    GreyWolf,
    WhaleOptimization,
    # ... and 50+ more algorithms
)
```

## Common Interface

All swarm intelligence algorithms inherit from `AbstractOptimizer` and implement:

```python
class SwarmAlgorithm(AbstractOptimizer):
    def __init__(
        self,
        func: Callable,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            func: Objective function to minimize
            lower_bound: Lower bound for all dimensions
            upper_bound: Upper bound for all dimensions
            dim: Problem dimensionality
            max_iter: Maximum iterations
            population_size: Number of agents (algorithm-specific default if None)
        """
        pass

    def search(self) -> tuple[np.ndarray, float]:
        """
        Execute optimization.

        Returns:
            Tuple of (best_solution, best_fitness)
        """
        pass
```

## Available Algorithms

See the [Swarm Intelligence algorithms section](/algorithms/swarm-intelligence/) for detailed documentation of each algorithm.

## Example Usage

```python
from opt.swarm_intelligence import ParticleSwarm, GreyWolf
from opt.benchmark.functions import shifted_ackley
import numpy as np

# Particle Swarm Optimization
pso = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=10,
    max_iter=100,
    population_size=30
)
solution, fitness = pso.search()
print(f"PSO - Best fitness: {fitness:.6e}")

# Grey Wolf Optimizer
gwo = GreyWolf(
    func=shifted_ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=10,
    max_iter=100,
    population_size=30
)
solution, fitness = gwo.search()
print(f"GWO - Best fitness: {fitness:.6e}")
```

## See Also

- [Abstract Optimizer](./abstract-optimizer) - Base class documentation
- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/) - Algorithm details
