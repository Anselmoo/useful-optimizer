# Classical Optimization Algorithms

Well-established optimization methods with proven convergence properties.

## Overview

Classical optimization algorithms include direct search methods, quasi-Newton methods, and derivative-free techniques that have been studied extensively in optimization theory.

## Available Algorithms

- [BFGS](./bfgs) - Broyden–Fletcher–Goldfarb–Shanno algorithm
- [Nelder-Mead](./nelder-mead) - Simplex-based derivative-free method
- [Simulated Annealing](./simulated-annealing) - Thermodynamics-inspired probabilistic technique
- [Hill Climbing](./hill-climbing) - Local search algorithm
- [Powell's Method](./powell) - Conjugate direction method
- [Trust Region](./trust-region) - Constrained optimization approach

## Usage Example

```python
from opt.classical import NelderMead, SimulatedAnnealing
from opt.benchmark.functions import rosenbrock

# Nelder-Mead
nm = NelderMead(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=10,
    dim=10,
    max_iter=1000
)
best_solution, best_fitness = nm.search()
```

## See Also

- [API Reference](/api/classical) - Complete API documentation
- [Gradient-Based Methods](../gradient-based/) - For gradient-based classical methods
