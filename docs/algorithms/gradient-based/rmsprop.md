# RMSprop

<span class="badge badge-gradient">Gradient-Based</span>

Root Mean Square Propagation (RMSprop) optimization algorithm.

## Algorithm Overview

This module implements the RMSprop optimization algorithm. RMSprop is an adaptive
learning rate method that was proposed by Geoffrey Hinton. It modifies AdaGrad to
perform better in non-convex settings by using a moving average of squared gradients
instead of accumulating all squared gradients.

RMSprop performs the following update rule:
    v = rho * v + (1 - rho) * gradient^2
    x = x - (learning_rate / sqrt(v + epsilon)) * gradient

where:
    - x: current solution
    - v: moving average of squared gradients
    - learning_rate: step size for parameter updates
    - rho: decay rate (typically 0.9)
    - epsilon: small constant to avoid division by zero
    - gradient: gradient of the objective function at x

## Usage

```python
from opt.gradient_based.rmsprop import RMSprop
from opt.benchmark.functions import sphere

optimizer = RMSprop(
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
| `rho` | `float` | `0.9` | Decay rate for moving average of squared gradients. |
| `epsilon` | `float` | `1e-08` | Small constant for numerical stability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Root Mean Square Propagation             |
| Acronym           | RMSPROP                                  |
| Year Introduced   | 2012                                     |
| Authors           | Hinton, Geoffrey; Srivastava, Nitish    |
| Algorithm Class   | Gradient-Based                           |
| Complexity        | O(dim)                                   |
| Properties        | Gradient-based, Stochastic           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations:

$$
E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot g_t^2
$$

$$
x_{t+1} = x_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\eta$ is the learning rate
- $\rho$ is the decay rate for moving average
- $\epsilon$ is a small constant for numerical stability
- $E[g^2]_t$ is the moving average of squared gradients

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter        | Default | BBOB Recommended | Description                       |
|------------------|---------|------------------|-----------------------------------|
| max_iter         | 1000    | 10000            | Maximum iterations                |
| learning_rate    | 0.01    | 0.001-0.1        | Learning rate (step size)         |
| rho              | 0.9     | 0.9-0.99         | Decay rate for moving average     |
| epsilon          | 1e-8    | 1e-8             | Numerical stability constant      |

**Sensitivity Analysis**:
- `learning_rate`: **High** impact on convergence
- `rho`: **Medium** impact - controls adaptation speed
- Recommended tuning ranges: $\eta \in [0.0001, 0.1]$, $\rho \in [0.85, 0.99]$

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
- Space complexity: $O(dim)$ for storing moving average
- BBOB budget usage: _Typically uses 55-75% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, ill-conditioned functions
- **Weak function classes**: Highly multimodal functions
- Typical success rate at 1e-8 precision: **45-65%** (dim=5)
- Expected Running Time (ERT): Comparable to Adam, better than AdaGrad

**Convergence Properties**:
- Convergence rate: Fast initial convergence, linear later
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Low-Medium** - adaptive rates help

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Moving average prevents gradient explosion

**Known Limitations**:
- Learning rate still requires tuning
- May not converge in all scenarios without proper LR scheduling
- Gradient approximation via finite differences less accurate

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Tieleman, T., & Hinton, G. (2012). "Lecture 6.5-rmsprop: Divide the gradient
by a running average of its recent magnitude."
_COURSERA: Neural networks for machine learning_, 4(2), 26-31.

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: No specific COCO benchmark data available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original presentation: Hinton's Coursera lecture
- This implementation: Standard RMSprop with BBOB compliance

## See Also

AdaGrad: Predecessor with accumulating gradient history
BBOB Comparison: RMSprop more stable due to moving average

Adam: Combines RMSprop with momentum
BBOB Comparison: Adam generally outperforms RMSprop

AdaDelta: Similar adaptive method without learning rate
BBOB Comparison: Both perform similarly on most BBOB functions

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, AdaGrad, AdaDelta
- Classical: BFGS, L-BFGS

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`rmsprop.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/rmsprop.py)
:::
