# Physics-Inspired Algorithms

Optimization algorithms based on physical phenomena and laws of physics.

## Overview

Physics-inspired algorithms mimic natural physical processes such as electromagnetic fields, gravitational forces, thermal dynamics, and atomic interactions to search for optimal solutions.

## Available Algorithms

- [Atom Search Optimization](./atom) - Based on atomic molecular dynamics
- [Equilibrium Optimizer](./equilibrium) - Inspired by thermodynamic equilibrium
- [Gravitational Search Algorithm](./gravitational) - Uses law of gravity and mass interactions
- [Charged System Search](./charged-system) - Based on Coulomb and Gauss laws

## Characteristics

- Model physical laws mathematically
- Use force, energy, or equilibrium concepts
- Often involve particle interactions
- Balance exploration and exploitation through physical analogies

## Usage Example

```python
from opt.physics_inspired import AtomSearch
from opt.benchmark.functions import sphere

optimizer = AtomSearch(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=100
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [API Reference](/api/) - Complete API documentation
- [Swarm Intelligence](../swarm-intelligence/) - Nature-inspired algorithms
