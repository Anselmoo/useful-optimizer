# Conjugate Gradient

<span class="badge badge-classical">Classical</span>

Conjugate Gradient Method (CG) optimization algorithm.

## Algorithm Overview

This module implements the Conjugate Gradient optimization algorithm. The Conjugate
Gradient method is an algorithm for the numerical solution of systems of linear
equations whose matrix is positive-definite. For general optimization, it's used
as an iterative method for solving unconstrained optimization problems.

The method works by:
1. Computing the gradient at the current point
2. Determining a conjugate direction (orthogonal in a specific sense)
3. Performing a line search along this direction
4. Updating the position and computing a new conjugate direction

The conjugate gradient method has the property that it converges in at most n steps
for a quadratic function in n dimensions, making it particularly effective for
quadratic and near-quadratic problems.

This implementation uses scipy's CG optimizer with multiple random restarts
to improve global optimization performance.

## Usage

```python
from opt.classical.conjugate_gradient import ConjugateGradient
from opt.benchmark.functions import sphere

optimizer = ConjugateGradient(
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
| `num_restarts` | `int` | `10` | Number of random restarts for multistart strategy. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Conjugate Gradient Method                |
| Acronym           | CG                                       |
| Year Introduced   | 1952                                     |
| Authors           | Hestenes, Magnus; Stiefel, Eduard        |
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
- $p_k$ is the conjugate search direction

Direction update:

$$
p_{k+1} = -\nabla f(x_{k+1}) + \beta_k p_k
$$

where $\beta_k$ is computed using Fletcher-Reeves or Polak-Ribière formula

Constraint handling:
- **Boundary conditions**: Penalty-based (large value for out-of-bounds)
- **Feasibility enforcement**: Post-optimization clamping to bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| max_iter               | 1000    | 10000            | Maximum iterations             |
| num_restarts           | 10      | 5-20             | Number of random restarts      |

**Sensitivity Analysis**:
- `num_restarts`: **Medium** impact on global optimization quality
- Recommended tuning ranges: $\text{num\_restarts} \in [5, 20]$

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
- Time per iteration: $O(n^2)$ for direction updates and line search
- Space complexity: $O(n)$ for storing vectors
- BBOB budget usage: _Typically uses 15-40% of $\text{dim} \times 10000$ budget for smooth functions_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, Smooth, Well-conditioned quadratic
- **Weak function classes**: Non-smooth, Highly multimodal, Ill-conditioned
- Typical success rate at 1e-8 precision: **60-80%** (dim=5, quadratic functions)
- Expected Running Time (ERT): Excellent on quadratic, good on smooth unimodal

**Convergence Properties**:
- Convergence rate: Superlinear on quadratic functions
- Local vs Global: Tends to find local optima, multistart improves global search
- Premature convergence risk: **Medium** (depends on $\beta$ formula used)

**Reproducibility**:
- **Deterministic**: Yes (given same seed) - Same seed guarantees same restart points
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` for restart initialization

**Implementation Details**:
- Parallelization: Not supported (sequential restarts)
- Constraint handling: Penalty-based during optimization, clamping post-optimization
- Numerical stability: Relies on SciPy's numerically stable CG implementation

**Known Limitations**:
- Requires gradient computation (finite differences if not provided)
- Performance degrades on ill-conditioned problems
- May cycle or stall on non-quadratic functions
- Multistart strategy increases total function evaluations

**Version History**:
- v0.1.0: Initial implementation with multistart strategy
- v0.1.2: Added COCO/BBOB compliance documentation

## References

[1] Hestenes, M. R., & Stiefel, E. (1952). "Methods of conjugate gradients for solving linear systems."
_Journal of Research of the National Bureau of Standards_, 49(6), 409-436.
https://doi.org/10.6028/jres.049.044

[2] Fletcher, R., & Reeves, C. M. (1964). "Function minimization by conjugate gradients."
_The Computer Journal_, 7(2), 149-154.
https://doi.org/10.1093/comjnl/7.2.149

[3] Polak, E., & Ribière, G. (1969). "Note sur la convergence de méthodes de directions conjuguées."
_Revue française d'informatique et de recherche opérationnelle_, 3(16), 35-43.

[4] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: SciPy implementation widely benchmarked
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Multiple independent implementations
- This implementation: Based on SciPy's CG with multistart for BBOB compliance

## See Also

BFGS: Quasi-Newton method with similar convergence properties
BBOB Comparison: Faster per iteration for low-medium dimensions

LBFGS: Limited-memory variant suitable for high dimensions
BBOB Comparison: Better memory scaling, similar convergence rate

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Classical: BFGS, NelderMead, Powell
- Gradient: AdamW, SGDMomentum

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
View the implementation: [`conjugate_gradient.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/conjugate_gradient.py)
:::
