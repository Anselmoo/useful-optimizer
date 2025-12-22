# Quick Start

This guide will get you up and running with Useful Optimizer in minutes.

## Basic Usage

### 1. Import an Optimizer

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley
```

### 2. Create the Optimizer

```python
optimizer = ParticleSwarm(
    func=shifted_ackley,       # Objective function to minimize
    lower_bound=-12.768,       # Lower bound of search space
    upper_bound=12.768,        # Upper bound of search space
    dim=2,                     # Number of dimensions
    max_iter=100               # Maximum iterations
)
```

### 3. Run the Optimization

```python
best_solution, best_fitness = optimizer.search()

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Complete Example

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import rosenbrock

import numpy as np

# Define the problem
optimizer = ParticleSwarm(
    func=rosenbrock,
    lower_bound=-5.0,
    upper_bound=10.0,
    dim=10,
    max_iter=500,
    population_size=50,      # Number of particles
    c1=2.0,                  # Cognitive parameter
    c2=2.0                   # Social parameter
)

# Run optimization
best_solution, best_fitness = optimizer.search()

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.6e}")
```

## Trying Different Algorithms

### Swarm Intelligence

```python
from opt.swarm_intelligence import (
    ParticleSwarm,
    AntColony,
    FireflyAlgorithm,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm
)
```

### Evolutionary Algorithms

```python
from opt.evolutionary import (
    GeneticAlgorithm,
    DifferentialEvolution,
    CMAES
)
```

### Gradient-Based

```python
from opt.gradient_based import (
    SGDMomentum,
    AdamW,
    RMSprop
)
```

### Classical Methods

```python
from opt.classical import (
    BFGS,
    NelderMead,
    SimulatedAnnealing
)
```

## Custom Objective Functions

You can optimize any function that takes a numpy array and returns a scalar:

```python
import numpy as np
from opt.swarm_intelligence import ParticleSwarm

# Define your own objective function
def my_function(x: np.ndarray) -> float:
    """Custom objective: sum of squares with a twist."""
    return np.sum(x**2) + 10 * np.sin(np.sum(x))

# Optimize it
optimizer = ParticleSwarm(
    func=my_function,
    lower_bound=-10.0,
    upper_bound=10.0,
    dim=5,
    max_iter=100
)

best_solution, best_fitness = optimizer.search()
```

## Using the Demo Utility

Every optimizer can be quickly tested with the built-in demo:

```python
from opt.demo import run_demo
from opt.swarm_intelligence import ParticleSwarm

# Run with default settings
run_demo(ParticleSwarm)

# Or customize parameters
run_demo(
    ParticleSwarm,
    max_iter=200,
    population_size=50,
    dim=10
)
```

### Command Line Demos

Run demos directly from the command line:

```bash
python -m opt.swarm_intelligence.particle_swarm
python -m opt.gradient_based.adamw
python -m opt.classical.simulated_annealing
```

## Next Steps

- **[Advanced Usage](./advanced)** - Learn about history tracking, custom constraints, and more
- **[Algorithm Reference](/algorithms/)** - Explore all 54+ algorithms
- **[Benchmarks](/benchmarks/)** - See performance comparisons
