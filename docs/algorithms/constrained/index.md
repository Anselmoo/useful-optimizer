# Constrained Optimization Algorithms

Algorithms designed to handle optimization problems with constraints.

## Overview

Constrained optimization algorithms can handle problems with equality and inequality constraints, bounds, and other restrictions on the solution space.

## Available Algorithms

- [Penalty Method](./penalty) - Converts constrained to unconstrained problem
- [Barrier Method](./barrier) - Interior point approach
- [Augmented Lagrangian](./augmented-lagrangian) - Lagrange multiplier method
- [Sequential Quadratic Programming](./sqp) - Iterative quadratic approximation
- [Trust Region Constrained](./trust-region-constrained) - Constrained trust region

## Usage Example

```python
from opt.constrained import PenaltyMethod
from opt.benchmark.functions import sphere

def constraint_eq(x):
    return x[0] + x[1] - 1.0  # x[0] + x[1] = 1

def constraint_ineq(x):
    return x[0] - 0.5  # x[0] >= 0.5

optimizer = PenaltyMethod(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    constraints_eq=[constraint_eq],
    constraints_ineq=[constraint_ineq]
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [API Reference](/api/constrained) - Complete API documentation
