# 🔒 Constrained Algorithms

Optimization with constraints.

This category contains **5 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Augmented Lagrangian Method](./augmented-lagrangian-method) | ALM | 1969 | Augmented Lagrangian Method (ALM) optimization algorithm. |
| [Barrier Method (Interior Point)](./barrier-method) | IPM | 1968 | Barrier Method (Interior Point Method) optimization algorithm. |
| [Penalty Method (Quadratic Penalty)](./penalty-method) | PM | 1943 | Penalty Method for constrained optimization. |
| [Sequential Quadratic Programming](./sequential-quadratic-programming) | SQP | 1963 | Sequential Quadratic Programming (SQP) optimization algorithm. |
| [Successive Linear Programming](./successive-linear-programming) | SLP | 1961 | Successive Linear Programming (SLP) optimization algorithm. |

## Usage

Every optimizer shares the same interface:

```python
from opt.constrained.augmented_lagrangian_method import AugmentedLagrangian
from opt.benchmark.functions import sphere

optimizer = AugmentedLagrangian(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)
