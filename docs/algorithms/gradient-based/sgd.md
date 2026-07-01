# Stochastic Gradient Descent

<span class="badge badge-gradient">Gradient-Based</span>

Stochastic Gradient Descent (SGD) optimization algorithm.

## Algorithm Overview

This module implements the Stochastic Gradient Descent optimization algorithm. SGD is
a gradient-based optimization algorithm that updates parameters in the direction
opposite to the gradient of the objective function. It is one of the most fundamental
and widely-used optimization algorithms in machine learning.

SGD performs the following update rule:
    x = x - learning_rate * gradient

where:
    - x: current solution
    - learning_rate: step size for parameter updates
    - gradient: gradient of the objective function at x

## Usage

```python
from opt.gradient_based.stochastic_gradient_descent import SGD
from opt.benchmark.functions import sphere

optimizer = SGD(
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
| `learning_rate` | `float` | `0.01` | Learning rate (step size). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Stochastic Gradient Descent              |
| Acronym           | SGD                                      |
| Year Introduced   | 1951                                     |
| Authors           | Robbins, Herbert; Monro, Sutton          |
| Algorithm Class   | Gradient-Based                           |
| Complexity        | O(dim)                                   |
| Properties        | Gradient-based, Stochastic           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equation:

$$
x_{t+1} = x_t - \eta \cdot g_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\eta$ is the learning rate

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter        | Default | BBOB Recommended | Description                       |
|------------------|---------|------------------|-----------------------------------|
| max_iter         | 1000    | 10000            | Maximum iterations                |
| learning_rate    | 0.01    | 0.001-0.1        | Learning rate (step size)         |

**Sensitivity Analysis**:
- `learning_rate`: **Very High** impact on convergence - most critical parameter
- Recommended tuning ranges: $\eta \in [0.0001, 0.1]$ (problem-dependent)

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
- Time per iteration: $O(dim)$ for gradient computation
- Space complexity: $O(dim)$ for solution storage
- BBOB budget usage: _Often uses full dim*10000 budget without convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Convex, well-conditioned functions
- **Weak function classes**: Ill-conditioned, multimodal functions
- Typical success rate at 1e-8 precision: **20-40%** (dim=5)
- Expected Running Time (ERT): Generally slower than adaptive methods

**Convergence Properties**:
- Convergence rate: Sublinear for convex functions
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Medium** - depends heavily on learning rate

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: No special provisions (vanilla SGD)

**Known Limitations**:
- Learning rate requires careful manual tuning
- No adaptive learning rate - single LR for all parameters
- Oscillates in ravines and valleys
- Slow convergence on ill-conditioned problems
- Not recommended for complex optimization without momentum

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Robbins, H., & Monro, S. (1951). "A Stochastic Approximation Method."
_The Annals of Mathematical Statistics_, 22(3), 400-407.
https://doi.org/10.1214/aoms/1177729586

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: No specific COCO benchmark data available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper: Classical algorithm, widely implemented
- This implementation: Standard SGD with BBOB compliance

## See Also

SGDMomentum: SGD with momentum term for acceleration
BBOB Comparison: Momentum variant typically converges faster

Adam: Adaptive learning rate method combining SGD ideas
BBOB Comparison: Adam generally outperforms vanilla SGD

RMSprop: Adaptive learning rate variant
BBOB Comparison: More stable than SGD on ill-conditioned problems

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, RMSprop, SGDMomentum
- Classical: BFGS, L-BFGS

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`stochastic_gradient_descent.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/stochastic_gradient_descent.py)
:::
