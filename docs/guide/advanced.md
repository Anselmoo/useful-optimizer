# Advanced Usage

This guide covers advanced features and customization options.

## History Tracking

Track the optimization progress over iterations:

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import rosenbrock

optimizer = ParticleSwarm(
    func=rosenbrock,
    lower_bound=-5.0,
    upper_bound=10.0,
    dim=10,
    max_iter=100,
    track_history=True  # Enable history tracking
)

best_solution, best_fitness = optimizer.search()

# Access convergence history
if hasattr(optimizer, 'best_fitness_history'):
    import matplotlib.pyplot as plt

    plt.plot(optimizer.best_fitness_history)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Curve')
    plt.yscale('log')
    plt.show()
```

## Algorithm-Specific Parameters

### Particle Swarm Optimization

```python
from opt.swarm_intelligence import ParticleSwarm

optimizer = ParticleSwarm(
    func=objective,
    lower_bound=-10.0,
    upper_bound=10.0,
    dim=10,
    max_iter=500,
    population_size=100,    # Number of particles
    w=0.7,                  # Inertia weight
    c1=1.5,                 # Cognitive coefficient
    c2=1.5                  # Social coefficient
)
```

### Differential Evolution

```python
from opt.evolutionary import DifferentialEvolution

optimizer = DifferentialEvolution(
    func=objective,
    lower_bound=-10.0,
    upper_bound=10.0,
    dim=10,
    max_iter=500,
    population_size=100,
    mutation_factor=0.8,    # F parameter
    crossover_rate=0.9      # CR parameter
)
```

### Adam Optimizer

```python
from opt.gradient_based import AdamW

optimizer = AdamW(
    func=objective,
    lower_bound=-10.0,
    upper_bound=10.0,
    dim=10,
    max_iter=1000,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01
)
```

## Comparing Multiple Algorithms

```python
import numpy as np
from opt.swarm_intelligence import ParticleSwarm, GreyWolfOptimizer
from opt.evolutionary import DifferentialEvolution
from opt.classical import SimulatedAnnealing
from opt.benchmark.functions import rosenbrock

# Define algorithms to compare
algorithms = {
    'PSO': ParticleSwarm,
    'GWO': GreyWolfOptimizer,
    'DE': DifferentialEvolution,
    'SA': SimulatedAnnealing
}

# Common parameters
params = {
    'func': rosenbrock,
    'lower_bound': -5.0,
    'upper_bound': 10.0,
    'dim': 10,
    'max_iter': 100
}

# Run comparison
results = {}
for name, AlgorithmClass in algorithms.items():
    optimizer = AlgorithmClass(**params)
    _, fitness = optimizer.search()
    results[name] = fitness
    print(f"{name}: {fitness:.6e}")
```

## Statistical Benchmarking

For rigorous algorithm comparison, run multiple independent runs:

```python
import numpy as np
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

n_runs = 30
fitness_values = []

for run in range(n_runs):
    np.random.seed(42 + run)  # Different seed per run

    optimizer = ParticleSwarm(
        func=shifted_ackley,
        lower_bound=-12.768,
        upper_bound=12.768,
        dim=10,
        max_iter=100
    )

    _, fitness = optimizer.search()
    fitness_values.append(fitness)

# Compute statistics
print(f"Mean: {np.mean(fitness_values):.6e}")
print(f"Std:  {np.std(fitness_values):.6e}")
print(f"Best: {np.min(fitness_values):.6e}")
print(f"Worst: {np.max(fitness_values):.6e}")
```

## Constrained Optimization

For problems with constraints, use the constrained optimization methods:

```python
from opt.constrained import AugmentedLagrangianMethod

def objective(x):
    return x[0]**2 + x[1]**2

def constraint1(x):
    """Inequality constraint: g(x) <= 0"""
    return x[0] + x[1] - 1  # x[0] + x[1] <= 1

optimizer = AugmentedLagrangianMethod(
    func=objective,
    lower_bound=-10.0,
    upper_bound=10.0,
    dim=2,
    max_iter=100,
    constraints=[constraint1]
)

best_solution, best_fitness = optimizer.search()
```

## Extending the Library

### Creating a Custom Optimizer

```python
from opt.abstract_optimizer import AbstractOptimizer
import numpy as np

class MyOptimizer(AbstractOptimizer):
    """Custom optimization algorithm."""

    def __init__(
        self,
        func,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 100,
        **kwargs
    ):
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            **kwargs
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Run the optimization."""
        # Initialize
        best_solution = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            self.dim
        )
        best_fitness = self.func(best_solution)

        # Main loop
        for _ in range(self.max_iter):
            # Your algorithm logic here
            candidate = best_solution + np.random.randn(self.dim) * 0.1
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)

            fitness = self.func(candidate)
            if fitness < best_fitness:
                best_solution = candidate
                best_fitness = fitness

        return best_solution, best_fitness
```

## Performance Tips

1. **Start with larger populations** for complex problems
2. **Use multiple restarts** for multi-modal functions
3. **Adjust learning rates** for gradient-based methods
4. **Enable history tracking** only when needed (memory overhead)
5. **Use appropriate bounds** - too wide reduces efficiency
