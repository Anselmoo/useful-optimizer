# Advanced Usage

This guide covers advanced features and patterns for using Useful Optimizer.

## Reproducible Results

For reproducible experiments, set a random seed:

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import rosenbrock

optimizer = ParticleSwarm(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=500,
    seed=42,  # Set seed for reproducibility
)

best_solution, best_fitness = optimizer.search()
```

## High-Dimensional Optimization

For problems with many dimensions, consider these tips:

```python
from opt.evolutionary import CMAES
from opt.benchmark.functions import sphere

# CMA-ES is excellent for high-dimensional problems
optimizer = CMAES(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=100,  # 100 dimensions
    population_size=50,
    max_iter=1000,
    sigma=1.0,  # Initial step size
)

best_solution, best_fitness = optimizer.search()
print(f"100D Sphere optimization: {best_fitness:.6e}")
```

## Constrained Optimization

For problems with constraints, use constrained optimization methods:

```python
from opt.constrained import AugmentedLagrangianMethod
import numpy as np

def objective(x):
    return x[0]**2 + x[1]**2

def constraint_eq(x):
    """Equality constraint: x[0] + x[1] = 1"""
    return [x[0] + x[1] - 1]

def constraint_ineq(x):
    """Inequality constraint: x[0] >= 0.2"""
    return [0.2 - x[0]]

optimizer = AugmentedLagrangianMethod(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    eq_constraints=constraint_eq,
    ineq_constraints=constraint_ineq,
    max_iter=500,
)

best_solution, best_fitness = optimizer.search()
print(f"Constrained solution: {best_solution}")
print(f"Constraint satisfaction: x[0] + x[1] = {sum(best_solution):.4f}")
```

## Multi-Objective Optimization

For problems with multiple objectives, use multi-objective optimizers:

```python
from opt.multi_objective import NSGAII
import numpy as np

def multi_objective(x):
    """Bi-objective function returning two objectives."""
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] - 1)**2 + (x[1] - 1)**2
    return np.array([f1, f2])

optimizer = NSGAII(
    func=multi_objective,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    n_objectives=2,
    population_size=100,
    max_iter=200,
)

pareto_front = optimizer.search()
print(f"Found {len(pareto_front)} Pareto optimal solutions")
```

## Bayesian Optimization

For expensive objective functions, use Bayesian optimization:

```python
from opt.probabilistic import BayesianOptimizer
import numpy as np

def expensive_function(x):
    """Simulating an expensive function evaluation."""
    return np.sin(x[0]) * np.cos(x[1]) + np.random.normal(0, 0.01)

optimizer = BayesianOptimizer(
    func=expensive_function,
    lower_bound=-np.pi,
    upper_bound=np.pi,
    dim=2,
    max_iter=50,  # Limited budget
    n_initial=10,  # Initial random samples
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution with {optimizer.max_iter} evaluations: {best_fitness:.6f}")
```

## Visualization

Enable visualization support:

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley
from opt.visualization import plot_convergence, plot_landscape

# Run with history tracking
optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    track_history=True,
)

best_solution, best_fitness = optimizer.search()

# Plot convergence curve
plot_convergence(optimizer.history['best_fitness'])

# Plot 2D landscape with search trajectory
plot_landscape(
    func=shifted_ackley,
    bounds=(-5, 5),
    trajectory=optimizer.history['best_solution'],
)
```

## Parameter Tuning

Different algorithms have different hyperparameters. Here are some guidelines:

### Particle Swarm Optimization

```python
from opt.swarm_intelligence import ParticleSwarm

optimizer = ParticleSwarm(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,        # More particles for harder problems
    max_iter=500,
    c1=2.0,                    # Cognitive coefficient
    c2=2.0,                    # Social coefficient
    w=0.7,                     # Inertia weight
    w_min=0.4,                 # Minimum inertia (for adaptive)
    w_max=0.9,                 # Maximum inertia (for adaptive)
)
```

### Differential Evolution

```python
from opt.evolutionary import DifferentialEvolution

optimizer = DifferentialEvolution(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=100,
    max_iter=500,
    F=0.8,                     # Mutation factor
    CR=0.9,                    # Crossover rate
    strategy='best/1/bin',     # Mutation strategy
)
```

### Adam Optimizer

```python
from opt.gradient_based import AdamW

optimizer = AdamW(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    max_iter=1000,
)
```

## Creating Custom Optimizers

Extend the base class to create your own optimizer:

```python
import numpy as np
from opt.abstract_optimizer import AbstractOptimizer

class MyCustomOptimizer(AbstractOptimizer):
    """A simple random search optimizer."""

    def search(self) -> tuple[np.ndarray, float]:
        """Perform random search optimization."""
        rng = np.random.default_rng(self.seed)

        best_solution = None
        best_fitness = float('inf')

        for iteration in range(self.max_iter):
            # Generate random candidate
            candidate = rng.uniform(
                self.lower_bound,
                self.upper_bound,
                size=self.dim
            )

            # Evaluate fitness
            fitness = self.func(candidate)

            # Update best
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = candidate.copy()

            # Track history if enabled
            if self.track_history:
                self.history['best_fitness'].append(best_fitness)
                self.history['best_solution'].append(best_solution)

        return best_solution, best_fitness

# Use your custom optimizer
optimizer = MyCustomOptimizer(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=1000,
)
best_solution, best_fitness = optimizer.search()
```

## Performance Tips

### Parallel Evaluation

For expensive objective functions, consider parallelization:

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def evaluate_parallel(func, candidates):
    """Evaluate multiple candidates in parallel."""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(func, candidates))
    return np.array(results)
```

### Warm Starting

Reuse previous solutions as starting points:

```python
# First optimization
optimizer1 = ParticleSwarm(func=objective, ...)
solution1, fitness1 = optimizer1.search()

# Second optimization with warm start
optimizer2 = ParticleSwarm(
    func=objective,
    ...,
    initial_solution=solution1,  # Use previous solution
)
solution2, fitness2 = optimizer2.search()
```

## Next Steps

- [API Reference](../api/index.md) - Complete API documentation
- [Benchmarks](../benchmarks/index.md) - View algorithm comparisons
- [Algorithm Categories](../algorithms/index.md) - Explore all algorithms
