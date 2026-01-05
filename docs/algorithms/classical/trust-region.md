# Trust Region

<span class="badge badge-classical">Classical</span>

Trust Region Optimizer.

## Algorithm Overview

This module implements Trust Region optimization algorithms. Trust region methods
are a class of optimization algorithms that work by defining a region around the
current point where a model (usually quadratic) of the objective function is trusted
to be accurate. The algorithm finds the step that minimizes the model within this
trust region.

Trust region methods have several advantages:
- They are globally convergent under reasonable assumptions
- They automatically adapt the step size based on the quality of the model
- They handle ill-conditioned problems better than line search methods
- They are robust to numerical difficulties

This implementation provides access to scipy's trust region methods including:
- trust-constr: Trust region method with constraints
- trust-exact: Trust region method with exact Hessian
- trust-krylov: Trust region method using Krylov subspace

## Usage

```python
from opt.classical.trust_region import TrustRegion
from opt.benchmark.functions import sphere

optimizer = TrustRegion(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | Required | Objective function to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `max_iter` | `int` | `1000` | Maximum iterations per restart. |
| `num_restarts` | `int` | `10` | Number of random restarts. |
| `method` | `str` | `'trust-constr'` | Trust region variant ('trust-constr', 'trust-exact', 'trust-krylov'). |
| `seed` | `int  \|  None` | `None` | Random seed for BBOB reproducibility. |

## See Also

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`trust_region.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/classical/trust_region.py)
:::
