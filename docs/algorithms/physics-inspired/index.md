---
title: Other Algorithms
description: Constrained optimization and probabilistic methods
tags:
  - constrained
  - probabilistic
---

# Other Algorithms

This section covers specialized algorithms for constrained optimization and probabilistic methods.

---

## Constrained Optimization

Methods for problems with explicit constraints:

| Algorithm | Description |
|-----------|-------------|
| **Augmented Lagrangian** | Converts constrained to unconstrained |
| **Successive Linear Programming** | Linear approximations |

### Example

```python
from opt.constrained import AugmentedLagrangianMethod
from opt.benchmark.functions import sphere

optimizer = AugmentedLagrangianMethod(
    func=sphere,
    lower_bound=-10,
    upper_bound=10,
    dim=2,
    max_iter=100,
)

best_solution, best_fitness = optimizer.search()
```

---

## Probabilistic Methods

Statistical approaches to optimization:

| Algorithm | Description |
|-----------|-------------|
| **Linear Discriminant Analysis** | Dimensionality reduction |
| **Parzen Tree Estimator** | TPE for hyperparameter optimization |

---

## When to Use

!!! success "Good For"
    - Problems with equality/inequality constraints
    - Hyperparameter optimization (TPE)
    - Bayesian optimization frameworks

!!! warning "Limitations"
    - Constrained methods may be complex to set up
    - Probabilistic methods can be computationally intensive
