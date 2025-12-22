# Benchmark Functions API

API reference for benchmark functions available in `opt.benchmark.functions`.

## Overview

All benchmark functions share the same interface:

```python
from opt.benchmark.functions import sphere, rosenbrock, ackley
import numpy as np

def benchmark_function(x: np.ndarray) -> float:
    """
    Args:
        x: Input vector of shape (n,)

    Returns:
        Scalar fitness value (lower is better)
    """
    pass
```

## Available Functions

For detailed mathematical definitions and properties, see [Benchmark Functions](/benchmarks/functions).

### Unimodal Functions

- `sphere(x)` - Simple sum of squares
- `rosenbrock(x)` - Rosenbrock valley function
- `schwefel_2_22(x)` - Schwefel 2.22 function
- `schwefel_1_2(x)` - Schwefel 1.2 function
- `step(x)` - Step function

### Multi-Modal Functions

- `ackley(x)` - Ackley function
- `shifted_ackley(x)` - Shifted Ackley function
- `rastrigin(x)` - Rastrigin function
- `griewank(x)` - Griewank function
- `levy(x)` - Levy function
- `schwefel(x)` - Schwefel function

## Usage

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import sphere, ackley

# Test with sphere function
optimizer = ParticleSwarm(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=100
)
best_solution, best_fitness = optimizer.search()

# Test with Ackley function
optimizer = ParticleSwarm(
    func=ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=10,
    max_iter=100
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [Benchmark Functions Details](/benchmarks/functions) - Mathematical definitions and properties
- [Benchmark Methodology](/benchmarks/methodology) - How benchmarks are conducted
- [Benchmark Results](/benchmarks/results) - Performance comparison results
