# Constrained Optimization API

API reference for constrained optimization algorithms in `opt.constrained`.

## Module Overview

```python
from opt.constrained import (
    PenaltyMethod,
    BarrierMethod,
    AugmentedLagrangian,
    SequentialQuadraticProgramming,
    TrustRegionConstrained,
)
```

## Common Interface

```python
class ConstrainedOptimizer(AbstractOptimizer):
    def __init__(
        self,
        func: Callable,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        constraints_eq: Optional[List[Callable]] = None,
        constraints_ineq: Optional[List[Callable]] = None,
        **kwargs
    ):
        pass

    def search(self) -> tuple[np.ndarray, float]:
        pass
```

## Constraint Specification

Constraints are specified as callable functions:

- **Equality constraints**: `g(x) = 0`
- **Inequality constraints**: `h(x) >= 0`

## Example Usage

```python
from opt.constrained import PenaltyMethod
from opt.benchmark.functions import sphere
import numpy as np

# Define constraints
def eq_constraint(x):
    return x[0] + x[1] - 1.0  # x[0] + x[1] = 1

def ineq_constraint(x):
    return x[0] - 0.5  # x[0] >= 0.5

# Penalty Method
penalty = PenaltyMethod(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    max_iter=100,
    constraints_eq=[eq_constraint],
    constraints_ineq=[ineq_constraint],
    penalty_factor=1000
)
solution, fitness = penalty.search()
print(f"Solution: {solution}")
print(f"Constraint satisfaction: eq={eq_constraint(solution):.6f}, ineq={ineq_constraint(solution):.6f}")
```

## See Also

- [Constrained Optimization](/algorithms/constrained/) - Algorithm details
