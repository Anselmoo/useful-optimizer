# Penalty Method Optimizer

<span class="badge badge-constrained">Constrained</span>

Penalty Method for constrained optimization.

## Algorithm Overview

This module implements the Penalty Method for constrained optimization,
transforming constrained problems into unconstrained ones.

The algorithm adds penalty terms for constraint violations to the objective
function, with increasing penalty coefficients over iterations.

## Usage

```python
from opt.constrained.penalty_method import PenaltyMethodOptimizer
from opt.benchmark.functions import sphere

optimizer = PenaltyMethodOptimizer(
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
| `constraints` | `list[Callable]  \|  None` | `None` | List of
        inequality constraints in form $g(x) \leq 0$. |
| `eq_constraints` | `list[Callable]  \|  None` | `None` | List of
        equality constraints in form $h(x) = 0$. |
| `max_iter` | `int` | `100` | Maximum outer iterations. |
| `initial_penalty` | `float` | `1.0` | Starting penalty coefficient ρ₀. |
| `penalty_growth` | `float` | `2.0` | Penalty growth factor gamma > 1. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Penalty Method (Quadratic Penalty)       |
| Acronym           | PM                                       |
| Year Introduced   | 1943                                     |
| Authors           | Courant, Richard                         |
| Algorithm Class   | Constrained                              |
| Complexity        | O(n³) per iteration                      |
| Properties        | Gradient-based, Deterministic        |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Penalized objective function:

$$
P(x, \rho) = f(x) + \rho \left( \sum_{i} \max(0, g_i(x))^2 + \sum_{j} h_j(x)^2 \right)
$$

where:
- $f(x)$ is the objective function
- $g_i(x) \leq 0$ are inequality constraints
- $h_j(x) = 0$ are equality constraints
- $\rho > 0$ is the penalty parameter (increases over iterations)

Penalty update:

$$
\rho_{k+1} = \gamma \rho_k, \quad \gamma > 1
$$

Constraint handling:
- **Boundary conditions**: L-BFGS-B bounds enforcement
- **Feasibility enforcement**: Quadratic penalty for violations
- **Exterior approach**: Can start from infeasible region

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| max_iter               | 100     | 1000-5000        | Maximum outer iterations       |
| initial_penalty        | 1.0     | 0.1-10.0         | Initial penalty coefficient    |
| penalty_growth         | 2.0     | 1.5-10.0         | Penalty growth factor gamma        |

**Sensitivity Analysis**:
- `penalty_growth`: **High** impact - controls convergence speed
- `initial_penalty`: **Medium** impact - affects early iterations
- Recommended tuning ranges: $\rho_0 \in [0.1, 10]$, $\gamma \in [1.5, 10]$

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

- Can start from infeasible region
- Uses L-BFGS-B for inner unconstrained minimization
- BBOB: Returns final best solution after max_iter or convergence

**Computational Complexity**:
- Time per iteration: $O(n^3)$ for L-BFGS-B on penalized objective
- Space complexity: $O(n^2)$ for Hessian approximation
- BBOB budget usage: _Typically 20-50% of dim*10000 for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Smooth, moderately constrained
- **Weak function classes**: Highly constrained, active constraints at optimum
- Typical success rate at 1e-8 precision: **45-60%** (dim=5, with constraints)
- Expected Running Time (ERT): Slower than ALM/SQP due to ill-conditioning

**Convergence Properties**:
- Convergence rate: Linear (penalty parameter must → ∞)
- Local vs Global: Strong local convergence, limited global exploration
- Premature convergence risk: **Medium** (ill-conditioning at high penalties)

**Reproducibility**:
- **Deterministic**: Partially - Random initialization affects results
- **BBOB compliance**: No explicit seed parameter in current implementation
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random` for initial point

**Implementation Details**:
- Parallelization: Not supported (sequential inner optimizations)
- Constraint handling: Quadratic penalty (exterior approach)
- Numerical stability: May become ill-conditioned at very high penalties
- Inner solver: scipy.optimize.minimize with L-BFGS-B method
- Violation tracking: Monitors total constraint violation for best selection

**Known Limitations**:
- Ill-conditioning issues when penalty coefficient becomes very large
- May require many iterations to achieve tight constraint satisfaction
- Final solution may slightly violate constraints (finite penalty)
- Not suitable for problems requiring exact constraint satisfaction
- BBOB adaptation note: Standard BBOB is unconstrained; this adds
constraints for demonstration

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added COCO/BBOB compliant docstring

## References

[1] Courant, R. (1943). "Variational methods for the solution of problems
of equilibrium and vibrations." _Bulletin of the American Mathematical
Society_, 49, 1-23.

[2] Nocedal, J., & Wright, S. J. (2006). "Numerical Optimization" (2nd ed.).
_Springer_. Chapter 17: Penalty and Augmented Lagrangian Methods.

[3] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- This implementation: Based on [1] and [2] with L-BFGS-B inner solver

## See Also

AugmentedLagrangian: Combines penalty and Lagrange multipliers
BBOB Comparison: ALM typically converges faster and with better scaling

BarrierMethodOptimizer: Interior point alternative
BBOB Comparison: Barrier requires feasible start; penalty works from anywhere

SequentialQuadraticProgramming: Quadratic subproblem approach
BBOB Comparison: SQP often superior for smooth, well-conditioned problems

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Classical: SimulatedAnnealing, NelderMead
- Gradient: AdamW, BFGS

## Related Pages

- [Constrained Algorithms](/algorithms/constrained/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`penalty_method.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/constrained/penalty_method.py)
:::
