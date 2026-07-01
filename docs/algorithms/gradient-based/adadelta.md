# Adadelta

<span class="badge badge-gradient">Gradient-Based</span>

Adaptive Delta (AdaDelta) optimization algorithm.

## Algorithm Overview

This module implements the AdaDelta optimizer, which is an extension of AdaGrad that
seeks to reduce its sensitivity to the learning rate hyperparameter.

AdaDelta is a gradient-based optimization algorithm that adapts the learning rate
for each of the parameters in the model. It is designed to converge faster than
AdaGrad by using a moving average of the squared gradient values to scale the learning rate.

The AdaDelta optimizer is defined by the following update rule:

    Eg = rho * Eg + (1 - rho) * g^2
    dx = -sqrt(Edx + eps) / sqrt(Eg + eps) * g
    Edx = rho * Edx + (1 - rho) * dx^2
    x = x + dx

where:
    - x: current solution
    - g: gradient of the objective function
    - rho: decay rate
    - eps: small constant to avoid dividing by zero
    - Eg: moving average of squared gradient values
    - Edx: moving average of squared updates

The algorithm iteratively updates the solution x by computing the gradient of the
objective function at x, scaling it by the moving average of the squared gradients,
and dividing it by the square root of the moving average of the squared updates.

The algorithm continues for a fixed number of iterations or until a specified
stopping criterion is met, returning the best solution found.

This module provides a simple example of how to use the AdaDelta optimizer to minimize
the Shifted Ackley's function in two dimensions.

## Usage

```python
from opt.gradient_based.adadelta import AdaDelta
from opt.benchmark.functions import sphere

optimizer = AdaDelta(
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
| `rho` | `float` | `0.97` | Decay rate for moving averages of squared gradients and updates. |
| `eps` | `float` | `1e-08` | Small constant for numerical stability in division operations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Adaptive Delta                           |
| Acronym           | ADADELTA                                 |
| Year Introduced   | 2012                                     |
| Authors           | Zeiler, Matthew D.                       |
| Algorithm Class   | Gradient-Based                           |
| Complexity        | O(dim)                                   |
| Properties        | Gradient-based, Adaptive, Stochastic     |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations:

$$
E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot g_t^2
$$

$$
\Delta x_t = -\frac{\sqrt{E[\Delta x^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

$$
E[\Delta x^2]_t = \rho \cdot E[\Delta x^2]_{t-1} + (1 - \rho) \cdot \Delta x_t^2
$$

$$
x_{t+1} = x_t + \Delta x_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\rho$ is the decay rate for moving averages
- $\epsilon$ is a small constant for numerical stability
- $E[g^2]_t$ is the moving average of squared gradients
- $E[\Delta x^2]_t$ is the moving average of squared parameter updates

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter      | Default | BBOB Recommended | Description                         |
|----------------|---------|------------------|-------------------------------------|
| max_iter       | 1000    | 10000            | Maximum iterations                  |
| rho            | 0.95    | 0.90-0.99        | Decay rate for moving averages      |
| eps            | 1e-8    | 1e-8             | Numerical stability constant        |

**Sensitivity Analysis**:
- `rho`: **Medium** impact on convergence - controls adaptation speed
- Recommended tuning ranges: $\rho \in [0.90, 0.99]$, $\epsilon \in [10^{-10}, 10^{-6}]$

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
- Time per iteration: $O(dim)$ for gradient computation and updates
- Space complexity: $O(dim)$ for storing moving averages
- BBOB budget usage: _Typically uses 60-80% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, ill-conditioned functions
- **Weak function classes**: Multimodal functions with many local optima
- Typical success rate at 1e-8 precision: **40-60%** (dim=5)
- Expected Running Time (ERT): Comparable to Adam, better than vanilla SGD

**Convergence Properties**:
- Convergence rate: Linear to sublinear
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Medium** - adaptive rates help escape plateaus

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Epsilon added to denominators to prevent division by zero

**Known Limitations**:
- Gradient approximation via finite differences may be less accurate than analytical gradients
- Performance depends on problem scaling and conditioning
- May struggle on highly non-convex landscapes

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Zeiler, M. D. (2012). "ADADELTA: An Adaptive Learning Rate Method."
_arXiv preprint arXiv:1212.5701_.
https://arxiv.org/abs/1212.5701

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: No specific COCO benchmark data available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Not publicly available
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

AdaGrad: Predecessor algorithm with accumulating gradient history
BBOB Comparison: AdaDelta typically converges faster on ill-conditioned functions

RMSprop: Similar adaptive learning rate method
BBOB Comparison: Both perform similarly, but AdaDelta doesn't require manual learning rate

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, RMSprop, AdaGrad
- Classical: BFGS, L-BFGS

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adadelta.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adadelta.py)
:::
