---
title: Quick Start
description: Get up and running with Useful Optimizer in minutes
---

# Quick Start

This guide will get you up and running with Useful Optimizer in just a few minutes.

---

## Your First Optimization

Let's optimize a simple mathematical function using Particle Swarm Optimization:

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

# Create the optimizer
optimizer = ParticleSwarm(
    func=shifted_ackley,      # Function to minimize
    lower_bound=-12.768,      # Lower bound of search space
    upper_bound=+12.768,      # Upper bound of search space
    dim=2,                    # Number of dimensions
    population_size=30,       # Number of particles
    max_iter=100,             # Maximum iterations
)

# Run the optimization
best_solution, best_fitness = optimizer.search()

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

!!! success "Expected Output"
    ```
    Best solution: [0.00123456, -0.00098765]
    Best fitness: 0.004523...
    ```

---

## Understanding the Components

### The Objective Function

Every optimization needs a function to minimize. Useful Optimizer includes several benchmark functions:

```python
from opt.benchmark.functions import (
    sphere,           # Simple quadratic function
    rosenbrock,       # Classic test function
    shifted_ackley,   # Multimodal function
    rastrigin,        # Highly multimodal
)
```

You can also define your own function:

```python
import numpy as np

def my_function(x):
    """A simple custom objective function."""
    return np.sum(x**2) + np.sin(np.sum(x))

optimizer = ParticleSwarm(
    func=my_function,
    lower_bound=-10,
    upper_bound=10,
    dim=5,
    max_iter=100,
)
best_solution, best_fitness = optimizer.search()
```

### The Optimizer

All optimizers follow a consistent interface:

```python
optimizer = OptimizerClass(
    func,              # Objective function
    lower_bound,       # Lower search boundary
    upper_bound,       # Upper search boundary
    dim,               # Problem dimensionality
    max_iter=100,      # Optional: max iterations
    # ... optimizer-specific parameters
)

best_solution, best_fitness = optimizer.search()
```

---

## Trying Different Algorithms

### Gradient-Based Optimization

For smooth, differentiable functions:

```python
from opt.gradient_based import AdamW
from opt.benchmark.functions import rosenbrock

optimizer = AdamW(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    learning_rate=0.01,
    max_iter=1000,
)
best_solution, best_fitness = optimizer.search()
```

### Classical Methods

Traditional optimization approaches:

```python
from opt.classical import BFGS
from opt.benchmark.functions import sphere

optimizer = BFGS(
    func=sphere,
    lower_bound=-10,
    upper_bound=10,
    dim=3,
    num_restarts=5,
)
best_solution, best_fitness = optimizer.search()
```

### Evolutionary Algorithms

Population-based evolution:

```python
from opt.evolutionary import DifferentialEvolution
from opt.benchmark.functions import shifted_ackley

optimizer = DifferentialEvolution(
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,
    max_iter=200,
)
best_solution, best_fitness = optimizer.search()
```

---

## Import Styles

Useful Optimizer supports multiple import styles:

=== "Categorical (Recommended)"

    ```python
    from opt.swarm_intelligence import ParticleSwarm
    from opt.gradient_based import AdamW
    from opt.classical import BFGS
    from opt.evolutionary import GeneticAlgorithm
    ```

=== "Direct"

    ```python
    from opt import ParticleSwarm, AdamW, BFGS, GeneticAlgorithm
    ```

---

## Running Built-in Demos

Every optimizer includes a demo you can run directly:

```bash
# Run PSO demo
python -m opt.swarm_intelligence.particle_swarm

# Run AdamW demo
python -m opt.gradient_based.adamw

# Run Simulated Annealing demo
python -m opt.classical.simulated_annealing
```

Or programmatically:

```python
from opt.demo import run_demo
from opt.swarm_intelligence import ParticleSwarm

# Run with default settings
run_demo(ParticleSwarm)

# Or customize
run_demo(
    ParticleSwarm,
    max_iter=200,
    population_size=50,
)
```

---

## Next Steps

Now that you've run your first optimization:

1. **Learn the concepts**: Read about [Basic Concepts](basic-concepts.md)
2. **Explore algorithms**: Browse the [Algorithm Documentation](../algorithms/index.md)
3. **Test with benchmarks**: Try different [Benchmark Functions](../benchmarks/functions.md)
