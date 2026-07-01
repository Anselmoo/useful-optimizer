# BFGS

<span class="badge badge-classical">Classical</span>

Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton optimization algorithm.

## Algorithm Overview

This module implements the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm.
BFGS is a quasi-Newton method that approximates Newton's method by using an approximation
to the inverse Hessian matrix. It's particularly effective for smooth optimization problems
and typically converges faster than first-order methods.

BFGS builds up an approximation to the inverse Hessian matrix using gradient information
from previous iterations. This makes it more efficient than computing the actual Hessian
while still providing second-order convergence properties.

This implementation uses scipy's BFGS optimizer with multiple random restarts to improve
global optimization performance.

## Usage

```python
from opt.classical.bfgs import BFGS
from opt.benchmark.functions import sphere

optimizer = BFGS(
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
| `num_restarts` | `int` | `25` | Number of random restarts for multistart strategy. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Broyden-Fletcher-Goldfarb-Shanno         |
| Acronym           | BFGS                                     |
| Year Introduced   | 1970                                     |
| Authors           | Broyden, Charles; Fletcher, Roger; Goldfarb, Donald; Shanno, David |
| Algorithm Class   | Classical                                |
| Complexity        | O(n²) per iteration                      |
| Properties        | Gradient-based, Deterministic        |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equation:

$$
x_{k+1} = x_k + \alpha_k p_k
$$

where:
- $x_k$ is the position at iteration $k$
- $\alpha_k$ is the step size from line search
- $p_k = -B_k^{-1} \nabla f(x_k)$ is the search direction
- $B_k$ is the approximation to the Hessian matrix

Hessian approximation update (BFGS formula):

$$
B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}
$$

where $s_k = x_{k+1} - x_k$ and $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$

Constraint handling:
- **Boundary conditions**: Penalty-based (large value for out-of-bounds)
- **Feasibility enforcement**: Post-optimization clamping to bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| max_iter               | 1000    | 10000            | Maximum iterations             |
| num_restarts           | 25      | 10-50            | Number of random restarts      |

**Sensitivity Analysis**:
- `num_restarts`: **High** impact on global optimization quality
- Recommended tuning ranges: $\text{num\_restarts} \in [10, 50]$

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
- Time per iteration: $O(n^2)$ for Hessian approximation update
- Space complexity: $O(n^2)$ for storing inverse Hessian approximation
- BBOB budget usage: _Typically uses 10-30% of $\text{dim} \times 10000$ budget for smooth functions_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, Smooth, Moderate conditioning
- **Weak function classes**: Non-smooth, Highly multimodal, Discontinuous
- Typical success rate at 1e-8 precision: **70-90%** (dim=5, smooth functions)
- Expected Running Time (ERT): Excellent on quadratic and near-quadratic functions

**Convergence Properties**:
- Convergence rate: Superlinear (quadratic near minimum for well-conditioned problems)
- Local vs Global: Strong local optimizer, multistart improves global search
- Premature convergence risk: **Low** for smooth functions, **High** for multimodal

**Reproducibility**:
- **Deterministic**: Yes (given same seed) - Same seed guarantees same restart points
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` for restart initialization

**Implementation Details**:
- Parallelization: Not supported (sequential restarts)
- Constraint handling: Penalty-based during optimization, clamping post-optimization
- Numerical stability: Relies on SciPy's numerically stable BFGS implementation

**Known Limitations**:
- Requires gradient computation (finite differences if not provided)
- Memory scales as O(n²), impractical for very high dimensions (>1000)
- Multistart strategy increases total function evaluations
- May converge to local minima without sufficient restarts

**Version History**:
- v0.1.0: Initial implementation with multistart strategy
- v0.1.2: Added COCO/BBOB compliance documentation

## References

[1] Broyden, C. G. (1970). "The Convergence of a Class of Double-rank Minimization Algorithms."
_IMA Journal of Applied Mathematics_, 6(1), 76-90.
https://doi.org/10.1093/imamat/6.1.76

[2] Fletcher, R. (1970). "A new approach to variable metric algorithms."
_The Computer Journal_, 13(3), 317-322.
https://doi.org/10.1093/comjnl/13.3.317

[3] Goldfarb, D. (1970). "A family of variable-metric methods derived by variational means."
_Mathematics of Computation_, 24(109), 23-26.
https://doi.org/10.1090/S0025-5718-1970-0258249-6

[4] Shanno, D. F. (1970). "Conditioning of quasi-Newton methods for function minimization."
_Mathematics of Computation_, 24(111), 647-656.
https://doi.org/10.1090/S0025-5718-1970-0274029-X

[5] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: SciPy implementation widely benchmarked
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Multiple independent implementations
- This implementation: Based on SciPy's BFGS with multistart for BBOB compliance

## See Also

LBFGS: Limited-memory variant with O(n) memory vs O(n²) for BFGS
BBOB Comparison: Similar convergence rate, better scaling for high dimensions

ConjugateGradient: First-order method without Hessian approximation
BBOB Comparison: Faster per iteration, slower convergence on ill-conditioned problems

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Classical: NelderMead, TrustRegion, Powell
- Gradient: AdamW, SGDMomentum
- Quasi-Newton: LBFGS

## Related Pages

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`bfgs.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/bfgs.py)
:::
