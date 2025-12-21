# Physics-Inspired Algorithms

Physics-inspired optimization algorithms are based on physical phenomena and laws. They simulate natural physical processes to solve optimization problems.

## Overview

| Property | Value |
|----------|-------|
| **Category** | Physics-Based |
| **Algorithms** | 4 |
| **Best For** | Continuous optimization |
| **Characteristic** | Physics law analogies |

## Algorithm List

### Gravitational Search Algorithm

Based on Newton's law of gravity and mass interactions.

```python
from opt.physics_inspired import GravitationalSearch

optimizer = GravitationalSearch(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,
    max_iter=500,
)
```

### Equilibrium Optimizer

Inspired by physical systems reaching equilibrium states.

```python
from opt.physics_inspired import EquilibriumOptimizer

optimizer = EquilibriumOptimizer(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=500,
)
```

### Complete Algorithm List

| Algorithm | Physical Phenomenon | Module |
|-----------|---------------------|--------|
| Atom Search | Atomic interactions | `atom_search` |
| Equilibrium Optimizer | System equilibrium | `equilibrium_optimizer` |
| Gravitational Search | Gravity law | `gravitational_search` |
| RIME Optimizer | Ice formation | `rime_optimizer` |

## See Also

- [API Reference: Physics-Inspired](../api/physics-inspired.md)
