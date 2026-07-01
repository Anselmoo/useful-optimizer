# Powell's Method

<span class="badge badge-classical">Classical</span>

Powell's Conjugate Direction Method optimization algorithm.

## Algorithm Overview

This module implements Powell's optimization algorithm. Powell's method is a
derivative-free optimization algorithm that performs sequential one-dimensional
minimizations along coordinate directions and then updates the search directions
based on the progress made.

Powell's method works by:
1. Starting with a set of linearly independent directions (usually coordinate axes)
2. Performing line searches along each direction
3. Replacing one of the directions with the overall direction of progress
4. Repeating until convergence

The method is particularly effective for functions that are not too irregular
and can handle functions where gradients are not available.

This implementation uses scipy's Powell optimizer with multiple random restarts
to improve global optimization performance.

## Usage

```python
from opt.classical.powell import Powell
from opt.benchmark.functions import sphere

optimizer = Powell(
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
| `seed` | `int  \|  None` | `None` | Random seed for BBOB reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Powell's Conjugate Direction Method      |
| Acronym           | POWELL                                   |
| Year Introduced   | 1964                                     |
| Authors           | Powell, Michael J. D.                    |
| Algorithm Class   | Classical                                |
| Complexity        | O(n²) per iteration                      |
| Properties        | Gradient-based, Deterministic        |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Sequential line searches along conjugate directions:

$$
x_{k+1} = x_k + \alpha_k d_k
$$

where:
- $x_k$ is the current position
- $\alpha_k$ is the optimal step size along direction $d_k$
- $d_k$ is the search direction (updated to maintain conjugacy)

Direction update strategy:
- Start with coordinate directions: $d_0, ..., d_{n-1} = e_0, ..., e_{n-1}$
- After $n$ line searches, replace one direction with overall progress direction
- New direction: $d_{new} = x_{n} - x_0$ (overall displacement)

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
- Time per iteration: $O(n^2)$
- Space complexity: $O(n^2)$
- BBOB budget usage: _20-50% of $\text{dim} \times 10000$_

**BBOB Performance Characteristics**:
- **Best function classes**: Smooth, Well-conditioned
- **Weak function classes**: Ill-conditioned, Discontinuous
- Success rate at 1e-8: **50-75%** (dim=5)

**Convergence Properties**:
- Convergence rate: Superlinear on quadratics
- Local vs Global: Local optimizer, multistart for global
- Premature convergence risk: **Medium**

**Reproducibility**:
- **Deterministic**: Yes (given same seed)
- **BBOB compliance**: seed required for 15 runs
- RNG: `numpy.random.default_rng(self.seed)`

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: COCO/BBOB compliance

## References

[1] Powell, M. J. D. (1964). "An efficient method for finding the minimum of a function of several variables without calculating derivatives."
_The Computer Journal_, 7(2), 155-162.
https://doi.org/10.1093/comjnl/7.2.155

[2] Hansen, N., Auger, A., et al. (2021). "COCO: A platform for comparing continuous optimizers."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

## See Also

NelderMead: Similar derivative-free simplex method
BBOB Comparison: Powell often faster on smooth functions
ConjugateGradient: Gradient-based variant of conjugate directions
BBOB Comparison: CG faster when gradients available

## Related Pages

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`powell.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/powell.py)
:::
