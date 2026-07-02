# Nelder-Mead

<span class="badge badge-classical">Classical</span>

Nelder-Mead Simplex Method optimization algorithm.

## Algorithm Overview

This module implements the Nelder-Mead optimization algorithm. Nelder-Mead is a
derivative-free optimization method that uses only function evaluations (no gradients).
It works by maintaining a simplex of n+1 points in n-dimensional space and iteratively
replacing the worst point with a better one through reflection, expansion, contraction,
and shrinkage operations.

The Nelder-Mead method is particularly useful for:
- Functions where gradients are not available or difficult to compute
- Noisy functions
- Functions with discontinuities
- Black-box optimization problems

This implementation uses scipy's Nelder-Mead optimizer with multiple random restarts
to improve global optimization performance.

## Usage

```python
from opt.classical.nelder_mead import NelderMead
from opt.benchmark.functions import sphere

optimizer = NelderMead(
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
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Nelder-Mead Simplex Method               |
| Acronym           | NM                                       |
| Year Introduced   | 1965                                     |
| Authors           | Nelder, John; Mead, Roger                |
| Algorithm Class   | Classical                                |
| Complexity        | $O((n+1) \times \text{evals})$ per iteration           |
| Properties        | Derivative-free, Deterministic       |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Simplex operations on $(n+1)$ vertices $\{x_0, ..., x_n\}$:

Reflection:

$$
x_r = \bar{x} + \alpha(\bar{x} - x_{n+1})
$$

Expansion:

$$
x_e = \bar{x} + \gamma(x_r - \bar{x})
$$

Contraction:

$$
x_c = \bar{x} + \rho(x_{n+1} - \bar{x})
$$

Shrinkage:

$$
x_i = x_0 + \sigma(x_i - x_0)
$$

where:
- $\bar{x}$ is the centroid of best $n$ points
- $x_{n+1}$ is the worst vertex
- $\alpha=1, \gamma=2, \rho=0.5, \sigma=0.5$ are standard coefficients

Constraint handling:
- **Boundary conditions**: Penalty-based (large value for out-of-bounds)
- **Feasibility enforcement**: Post-optimization clamping to bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| max_iter               | 1000    | 10000            | Maximum iterations             |
| num_restarts           | 25      | 10-50            | Number of random restarts      |

**Sensitivity Analysis**:
- `num_restarts`: **High** impact on finding global optimum
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
- Time per iteration: $O((n+1) \times f_{evals})$ for simplex operations
- Space complexity: $O(n^2)$ for storing simplex vertices
- BBOB budget usage: _Typically uses 30-60% of $\text{dim} \times 10000$ budget_

**BBOB Performance Characteristics**:
- **Best function classes**: Non-smooth, Low-dimensional, Noisy functions
- **Weak function classes**: High-dimensional, Ill-conditioned
- Typical success rate at 1e-8 precision: **40-70%** (dim=2-5)
- Expected Running Time (ERT): Good on low-dim, degrades rapidly with dimension

**Convergence Properties**:
- Convergence rate: Sub-linear in general (no quadratic convergence guarantee)
- Local vs Global: Tends to find local optima, multistart essential for global search
- Premature convergence risk: **High** without restarts (simplex can degenerate)

**Reproducibility**:
- **Deterministic**: Yes (given same seed) - Same seed guarantees same restart points
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` for restart initialization

**Implementation Details**:
- Parallelization: Not supported (sequential restarts)
- Constraint handling: Penalty-based during optimization, clamping post-optimization
- Numerical stability: Sensitive to simplex degeneracy, SciPy includes safeguards

**Known Limitations**:
- No gradient information used (slower than gradient methods on smooth functions)
- Performance degrades rapidly with dimension (curse of dimensionality)
- Simplex can collapse to subspace (degenerate simplex)
- Lacks convergence guarantees for non-convex functions

**Version History**:
- v0.1.0: Initial implementation with multistart strategy
- v0.1.2: Added COCO/BBOB compliance documentation

## References

[1] Nelder, J. A., & Mead, R. (1965). "A simplex method for function minimization."
_The Computer Journal_, 7(4), 308-313.
https://doi.org/10.1093/comjnl/7.4.308

[2] Wright, M. H. (1996). "Direct search methods: Once scorned, now respectable."
_Pitman Research Notes in Mathematics Series_, 191-208.

[3] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Classic benchmark standard, widely evaluated
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: FORTRAN implementations available
- This implementation: Based on SciPy's Nelder-Mead with multistart for BBOB compliance

## See Also

Powell: Another derivative-free method using conjugate directions
BBOB Comparison: Similar performance, Powell may converge faster on smooth functions

BFGS: Gradient-based alternative with faster convergence when gradients available
BBOB Comparison: BFGS faster on smooth functions, Nelder-Mead better for non-smooth

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Classical: Powell, HillClimbing, TabuSearch
- Gradient: BFGS, LBFGS, ConjugateGradient

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

Convergence, final-fitness distribution and performance profile on `rastrigin` (5D), averaged over independent runs (compared against representative baselines):

<ClientOnly>
  <BenchmarkCharts
    algorithm="NelderMead"
    functionName="rastrigin"
    :dimension="5"
    :compareWith="['GreyWolfOptimizer', 'ParticleSwarm', 'AntColony']"
  />
</ClientOnly>

## Related Pages

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`nelder_mead.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/nelder_mead.py)
:::
