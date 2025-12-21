# Constrained Optimization Algorithms

Constrained optimization algorithms handle problems with equality and inequality constraints.

## Overview

| Property | Value |
|----------|-------|
| **Category** | Constrained Methods |
| **Algorithms** | 5 |
| **Best For** | Problems with constraints |
| **Characteristic** | Constraint handling |

## Algorithm List

### Augmented Lagrangian Method

Transforms constrained problems into unconstrained ones.

```python
from opt.constrained import AugmentedLagrangianMethod

optimizer = AugmentedLagrangianMethod(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    eq_constraints=equality_func,
    ineq_constraints=inequality_func,
    max_iter=500,
)
```

### Barrier Method

Interior point method using barrier functions.

```python
from opt.constrained import BarrierMethod

optimizer = BarrierMethod(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    constraints=constraints,
    max_iter=500,
)
```

### Complete Algorithm List

| Algorithm | Method | Module |
|-----------|--------|--------|
| Augmented Lagrangian | Penalty + Lagrangian | `augmented_lagrangian_method` |
| Barrier Method | Interior point | `barrier_method` |
| Penalty Method | External penalties | `penalty_method` |
| Sequential Quadratic Programming | QP subproblems | `sequential_quadratic_programming` |
| Successive Linear Programming | LP approximations | `successive_linear_programming` |

## See Also

- [API Reference: Constrained](../api/constrained.md)
