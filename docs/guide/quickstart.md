# Quick Start

Get up and running with Useful Optimizer in just a few minutes!

## Basic Usage Pattern

All optimizers in Useful Optimizer follow a consistent pattern:

1. **Import** the optimizer and a benchmark function
2. **Create** an optimizer instance with parameters
3. **Call** the `search()` method
4. **Receive** the best solution and fitness

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

# Create optimizer
optimizer = ParticleSwarm(
    func=shifted_ackley,           # Objective function
    lower_bound=-12.768,           # Search space lower bound
    upper_bound=+12.768,           # Search space upper bound
    dim=2,                         # Problem dimensionality
    population_size=100,           # Number of particles
    max_iter=1000,                 # Maximum iterations
)

# Run optimization
best_solution, best_fitness = optimizer.search()

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Import Styles

Useful Optimizer supports two import styles:

=== "Categorical Imports (Recommended)"

    ```python
    from opt.swarm_intelligence import ParticleSwarm
    from opt.gradient_based import AdamW
    from opt.classical import BFGS
    from opt.evolutionary import GeneticAlgorithm
    from opt.metaheuristic import HarmonySearch
    ```

=== "Direct Imports"

    ```python
    from opt import ParticleSwarm, AdamW, BFGS, GeneticAlgorithm, HarmonySearch
    ```

## Custom Objective Functions

You can optimize any function that takes a NumPy array and returns a float:

```python
import numpy as np
from opt.swarm_intelligence import ParticleSwarm

# Define your own objective function
def my_function(x: np.ndarray) -> float:
    """Custom objective function to minimize."""
    return np.sum(x**2) + np.sin(np.sum(x)) * 10

# Optimize
optimizer = ParticleSwarm(
    func=my_function,
    lower_bound=-10,
    upper_bound=10,
    dim=5,
    max_iter=500,
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Comparing Optimizers

Easily compare different optimizers on the same problem:

```python
from opt.swarm_intelligence import ParticleSwarm, GreyWolfOptimizer
from opt.evolutionary import DifferentialEvolution
from opt.classical import SimulatedAnnealing
from opt.benchmark.functions import rosenbrock

# Common settings
settings = {
    "func": rosenbrock,
    "lower_bound": -5,
    "upper_bound": 5,
    "dim": 10,
    "max_iter": 500,
}

# Test multiple optimizers
optimizers = [
    ("PSO", ParticleSwarm(**settings)),
    ("GWO", GreyWolfOptimizer(**settings)),
    ("DE", DifferentialEvolution(**settings)),
    ("SA", SimulatedAnnealing(**settings)),
]

print("Optimizer Comparison")
print("-" * 40)
for name, opt in optimizers:
    solution, fitness = opt.search()
    print(f"{name:10} | Fitness: {fitness:.6e}")
```

## Using the Demo System

All optimizers include a standardized demo:

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
    c1=2.0,
    c2=2.0
)
```

Run demos from the command line:

```bash
python -m opt.swarm_intelligence.particle_swarm
python -m opt.gradient_based.adamw
python -m opt.classical.simulated_annealing
```

## Gradient-Based Optimizers

For differentiable objective functions, gradient-based optimizers are often more efficient:

```python
from opt.gradient_based import AdamW
from opt.benchmark.functions import sphere

optimizer = AdamW(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    learning_rate=0.01,
    weight_decay=0.01,
    max_iter=1000,
)

best_solution, best_fitness = optimizer.search()
print(f"Best fitness: {best_fitness:.10e}")
```

## Tracking Optimization History

Enable history tracking for visualization and analysis:

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    track_history=True,  # Enable tracking
)

best_solution, best_fitness = optimizer.search()

# Access history
history = optimizer.history
print(f"Iterations tracked: {len(history['best_fitness'])}")
print(f"Final fitness: {history['best_fitness'][-1]}")
```

## Next Steps

- [Advanced Usage](advanced.md) - Learn about advanced features
- [Algorithm Overview](../algorithms/index.md) - Explore all available algorithms
- [API Reference](../api/index.md) - Detailed API documentation
- [Benchmarks](../benchmarks/index.md) - View benchmark comparisons
