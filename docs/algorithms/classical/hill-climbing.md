# Hill Climbing

<span class="badge badge-classical">Classical</span>

Hill Climbing local search optimization algorithm.

## Algorithm Overview

This module implements the Hill Climbing optimizer, which performs a hill climbing
search to find the optimal solution for a given function within the specified bounds.

The HillClimbing class is the main class that implements the optimizer. It takes the
objective function, lower and upper bounds of the search space, dimensionality of the
search space, and other optional parameters as input. The search method performs the
hill climbing search and returns the optimal solution and its corresponding score.

Example usage:
    optimizer = HillClimbing(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        max_iter=5000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.classical.hill_climbing import HillClimbing
from opt.benchmark.functions import sphere

optimizer = HillClimbing(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `initial_step_sizes` | `float` | `1.0` | Initial step size for all dimensions. |
| `acceleration` | `float` | `1.2` | Factor for step size adaptation. |
| `epsilon` | `float` | `1e-06` | Convergence threshold for fitness change. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Hill Climbing                            |
| Acronym           | HC                                       |
| Year Introduced   | 1958                                     |
| Authors           | Various (classic heuristic method)       |
| Algorithm Class   | Classical                                |
| Complexity        | $O(n \times \text{candidates} \times \text{iterations})$           |
| Properties        | Single-solution, Deterministic       |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update for each dimension $i$:

$$
x_i^{t+1} = x_i^t + s_i^t \cdot \delta
$$

where:
- $x_i^t$ is position in dimension $i$ at iteration $t$
- $s_i^t$ is the adaptive step size for dimension $i$
- $\delta \in \{-a, -1/a, 1/a, a\}$ are candidate multipliers
- $a$ is the acceleration parameter

Step size adaptation:

$$
s_i^{t+1} = \begin{cases}
\delta & \text{if improvement found} \\
s_i^t / a & \text{otherwise (reduce step)}
\end{cases}
$$

Constraint handling:
- **Boundary conditions**: Implicit (function evaluation at boundary)
- **Feasibility enforcement**: Natural bounds from search process

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| max_iter               | 1000    | 5000-10000       | Maximum iterations             |
| initial_step_sizes     | 1.0     | 0.1-1.0          | Initial step size              |
| acceleration           | 1.2     | 1.1-1.5          | Step adaptation factor         |
| epsilon                | 1e-6    | 1e-8             | Convergence threshold          |

**Sensitivity Analysis**:
- `acceleration`: **High** impact on convergence speed and stability
- `initial_step_sizes`: **Medium** impact on exploration
- Recommended tuning: $a \in [1.1, 1.5]$, $s_0 \in [0.1, 1.0]$

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
- Time per iteration: $O(n \times c)$ where $c=4$ candidates per dimension
- Space complexity: $O(n)$ for storing position and step sizes
- BBOB budget usage: _Typically uses 20-50% of $\text{dim} \times 10000$ budget_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, Smooth, Low-dimensional
- **Weak function classes**: Multimodal (gets stuck in local optima)
- Typical success rate at 1e-8 precision: **30-60%** (dim=2-5, unimodal)
- Expected Running Time (ERT): Fast on unimodal, poor on multimodal

**Convergence Properties**:
- Convergence rate: Linear when far from optimum, can be fast initially
- Local vs Global: Pure local optimizer, no mechanism to escape local minima
- Premature convergence risk: **Very High** (guaranteed to get stuck in local optimum)

**Reproducibility**:
- **Deterministic**: Yes (given same seed) - Deterministic after initialization
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` for initialization only

**Implementation Details**:
- Parallelization: Not supported (inherently sequential coordinate descent)
- Constraint handling: Natural bounds (evaluates function at search points)
- Numerical stability: Step size reduction prevents infinite loops

**Known Limitations**:
- Cannot escape local optima (fundamental limitation of greedy search)
- Performance highly dependent on initialization
- Coordinate-wise search can be inefficient on rotated functions
- No global convergence guarantees

**Version History**:
- v0.1.0: Initial implementation with adaptive step sizes
- v0.1.2: Added COCO/BBOB compliance documentation

## References

[1] Russell, S. J., & Norvig, P. (2010). "Artificial Intelligence: A Modern Approach" (3rd ed.).
_Prentice Hall_, Chapter 4: Beyond Classical Search.

[2] Selman, B., & Gomes, C. P. (2006). "Hill-climbing search."
_Encyclopedia of Cognitive Science_.

[3] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Classic local search baseline
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original concept: Classic AI heuristic with many variants
- This implementation: Adaptive step size with acceleration-based exploration

## See Also

SimulatedAnnealing: Probabilistic variant that can escape local optima
BBOB Comparison: SA better on multimodal, HC faster on unimodal

TabuSearch: Memory-based local search avoiding recent solutions
BBOB Comparison: Tabu better exploration, HC simpler and faster

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Classical: SimulatedAnnealing, TabuSearch
- Local Search: All classical methods can be viewed as local search

## Related Pages

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`hill_climbing.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/hill_climbing.py)
:::
