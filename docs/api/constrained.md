# Constrained API

API reference for the 5 classes of `opt.constrained`. This page is generated from the source; see the [algorithm pages](/algorithms/constrained/) for formulations and references.

## Imports

```python
from opt.constrained import (
    AugmentedLagrangian,
    BarrierMethodOptimizer,
    PenaltyMethodOptimizer,
    SequentialQuadraticProgramming,
    SuccessiveLinearProgramming,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`AugmentedLagrangian`](/algorithms/constrained/augmented-lagrangian-method) | Augmented Lagrangian Method | `c=1`, `lambda_=0.1`, `static_cost=10000000000.0` |
| [`BarrierMethodOptimizer`](/algorithms/constrained/barrier-method) | Barrier Method (Interior Point) | `constraints=None`, `initial_mu=10.0`, `mu_reduction=0.5` |
| [`PenaltyMethodOptimizer`](/algorithms/constrained/penalty-method) | Penalty Method (Quadratic Penalty) | `constraints=None`, `eq_constraints=None`, `initial_penalty=1.0`, `penalty_growth=2.0` |
| [`SequentialQuadraticProgramming`](/algorithms/constrained/sequential-quadratic-programming) | Sequential Quadratic Programming | `constraints=None`, `eq_constraints=None`, `tol=1e-06` |
| [`SuccessiveLinearProgramming`](/algorithms/constrained/successive-linear-programming) | Successive Linear Programming | - |
