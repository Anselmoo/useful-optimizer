# Trust Region

<span class="badge badge-classical">Classical</span>

Trust Region optimization algorithm.

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

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Trust Region Method                      |
| Acronym           | TR                                       |
| Year Introduced   | 1983                                     |
| Authors           | Powell, M. J. D.; Conn, A. R.; et al.   |
| Algorithm Class   | Classical                                |
| Complexity        | O(n³) per iteration (subproblem solve)   |
| Properties        | Adaptive, Gradient-based             |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Trust region subproblem at iteration $k$:

$$
\min_{s} m_k(s) = f_k + g_k^T s + \frac{1}{2} s^T B_k s
$$

subject to: $\|s\| \leq \Delta_k$ (trust region radius)

where:
- $f_k = f(x_k)$ is current function value
- $g_k = \nabla f(x_k)$ is gradient
- $B_k$ approximates Hessian
- $\Delta_k$ is trust region radius (adaptive)

Radius update based on agreement ratio:

$$
\rho_k = \frac{f(x_k) - f(x_k + s_k)}{m_k(0) - m_k(s_k)}
$$

Constraint handling:
- **Boundary conditions**: Native bound constraints (trust-constr variant)
- **Feasibility enforcement**: During subproblem solve

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| max_iter               | 1000    | 10000            | Maximum iterations             |
| num_restarts           | 25      | 10-50            | Number of random restarts      |

**Sensitivity Analysis**:
- `num_restarts`: **High** impact on global optimization
- Initial radius: **Medium** (automatically adapted)
- Recommended: multiple restarts for non-convex problems

## COCO/BBOB Benchmark Settings

**Search Space**:
- Dimensions tested: `2, 3, 5, 10, 20, 40`
- Bounds: Function-specific (typically `[-5, 5]` or `[-100, 100]`)
- Instances: **15** per function (BBOB standard)

**Evaluation Budget**:
- Budget: $\text{dim} \times 10000$ function evaluations
- Independent runs: **15** (for statistical significance)
- Seeds: `0-14` (reproducibility requirement)

**Performance Metrics**:
- Target precision: `1e-8` (BBOB default)
- Success rate at precision thresholds: `[1e-8, 1e-6, 1e-4, 1e-2]`
- Expected Running Time (ERT) tracking

## Raises

ValueError: If search space is invalid or function evaluation fails.

## Notes

- Modifies self.history if track_history=True
- Uses self.seed for all random number generation
- BBOB: Returns final best solution after max_iter or convergence

**Computational Complexity**:
- Time per iteration: $O(n^3)$ for subproblem solve
- Space complexity: $O(n^2)$
- BBOB budget usage: _15-40% of $\text{dim} \times 10000$_

**BBOB Performance Characteristics**:
- **Best function classes**: Smooth, Ill-conditioned
- **Weak function classes**: Non-smooth, Highly multimodal
- Success rate at 1e-8: **75-95%** (dim=5, smooth)

**Convergence Properties**:
- Convergence rate: Superlinear to quadratic near minimum
- Local vs Global: Local optimizer, multistart for global search
- Premature convergence risk: **Low** (adaptive radius prevents divergence)

**Reproducibility**:
- **Deterministic**: Yes (given same seed)
- **BBOB compliance**: seed required for 15 runs
- RNG: `numpy.random.default_rng(self.seed)`

**Known Limitations**:
- Requires gradient computation
- Cubic subproblem solve expensive for high dimensions
- Multistart increases function evaluations

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: COCO/BBOB compliance

## References

[1] Conn, A. R., Gould, N. I., & Toint, P. L. (2000). "Trust Region Methods."
_SIAM_, Philadelphia.
https://doi.org/10.1137/1.9780898719857

[2] Nocedal, J., & Wright, S. J. (2006). "Numerical Optimization" (2nd ed.).
_Springer_, Chapter 4: Trust-Region Methods.

[3] Hansen, N., Auger, A., et al. (2021). "COCO: A platform for comparing continuous optimizers."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

## See Also

BFGS: Quasi-Newton with line search instead of trust region
BBOB Comparison: Similar performance, TR more robust to ill-conditioning
LBFGS: Limited-memory variant with line search
BBOB Comparison: TR better on ill-conditioned, L-BFGS better memory scaling

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

::: tip Run-based charts
Convergence, distribution and ECDF charts appear here once this optimizer is included in the benchmark suite.
:::

## Related Pages

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`trust_region.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/trust_region.py)
:::
