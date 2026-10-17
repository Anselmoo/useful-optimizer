# Physics-Inspired API

API reference for the 4 classes of `opt.physics_inspired`. This page is generated from the source; see the [algorithm pages](/algorithms/physics-inspired/) for formulations and references.

## Imports

```python
from opt.physics_inspired import (
    AtomSearchOptimizer,
    EquilibriumOptimizer,
    GravitationalSearchOptimizer,
    RIMEOptimizer,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`AtomSearchOptimizer`](/algorithms/physics-inspired/atom-search) | Atom Search Optimization | - |
| [`EquilibriumOptimizer`](/algorithms/physics-inspired/equilibrium-optimizer) | Equilibrium Optimizer | `a1=_A1`, `a2=_A2`, `gp=_GP` |
| [`GravitationalSearchOptimizer`](/algorithms/physics-inspired/gravitational-search) | Gravitational Search Algorithm | `g0=_GRAVITATIONAL_CONSTANT_INITIAL`, `alpha=_GRAVITATIONAL_DECAY_RATE` |
| [`RIMEOptimizer`](/algorithms/physics-inspired/rime-optimizer) | RIME: A Physics-based Optimization | - |
