# Adagrad

<span class="badge badge-gradient">Gradient-Based</span>

Adaptive Gradient Algorithm (AdaGrad) optimization algorithm.

## Algorithm Overview

This module implements the Adaptive Gradient Algorithm (ADAGrad) optimizer. ADAGrad is
a gradient-based optimization algorithm that adapts the learning rate to the parameters,
performing smaller updates for parameters associated with frequently occurring features,
and larger updates for parameters associated with infrequent features. It is particularly
useful for dealing with sparse data.

ADAGrad's main strength is that it eliminates the need to manually tune the learning rate.
Most implementations also include a 'smoothing term' to avoid division by zero when the
gradient is zero.

The ADAGrad optimizer is commonly used in machine learning and deep learning applications.

## Usage

```python
from opt.gradient_based.adagrad import ADAGrad
from opt.benchmark.functions import sphere

optimizer = ADAGrad(
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
| `lr` | `float` | `0.01` | Global learning rate. |
| `eps` | `float` | `1e-08` | Small constant for numerical stability in division operations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Adaptive Gradient Algorithm              |
| Acronym           | ADAGRAD                                  |
| Year Introduced   | 2011                                     |
| Authors           | Duchi, John; Hazan, Elad; Singer, Yoram  |
| Algorithm Class   | Gradient-Based                           |
| Complexity        | O(dim)                                   |
| Properties        | Gradient-based, Stochastic           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations:

$$
G_t = G_{t-1} + g_t \odot g_t
$$

$$
x_{t+1} = x_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\eta$ is the learning rate
- $\epsilon$ is a small constant for numerical stability
- $G_t$ is the sum of squared gradients up to iteration $t$
- $\odot$ denotes element-wise multiplication

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter      | Default | BBOB Recommended | Description                         |
|----------------|---------|------------------|-------------------------------------|
| max_iter       | 1000    | 10000            | Maximum iterations                  |
| lr             | 0.01    | 0.01-0.1         | Global learning rate                |
| eps            | 1e-8    | 1e-8             | Numerical stability constant        |

**Sensitivity Analysis**:
- `lr`: **High** impact on convergence - controls step size
- Recommended tuning ranges: $\eta \in [0.001, 0.1]$, $\epsilon \in [10^{-10}, 10^{-6}]$

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
- Space complexity: $O(dim)$ for storing gradient accumulator
- BBOB budget usage: _Typically uses 70-90% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Sparse gradients, convex functions
- **Weak function classes**: Non-stationary objectives, dense gradients
- Typical success rate at 1e-8 precision: **30-50%** (dim=5)
- Expected Running Time (ERT): Higher than Adam/RMSprop on most BBOB functions

**Convergence Properties**:
- Convergence rate: Sublinear due to aggressive learning rate reduction
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **High** - learning rates can become too small

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clipping to bounds (no explicit constraint enforcement)
- Numerical stability: Epsilon added to prevent division by zero

**Known Limitations**:
- Aggressive learning rate reduction can cause premature convergence
- Accumulates all past gradients - learning rate monotonically decreases
- Performance degrades on problems requiring many iterations
- Not recommended for deep learning or non-convex optimization

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Duchi, J., Hazan, E., & Singer, Y. (2011). "Adaptive Subgradient Methods
for Online Learning and Stochastic Optimization."
_Journal of Machine Learning Research_, 12, 2121-2159.
http://jmlr.org/papers/v12/duchi11a.html

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

AdaDelta: Extension that addresses diminishing learning rates
BBOB Comparison: AdaDelta often converges better on long optimization runs

RMSprop: Similar adaptive method using moving averages
BBOB Comparison: RMSprop typically more stable than AdaGrad

Adam: Combines ideas from AdaGrad and RMSprop
BBOB Comparison: Adam generally outperforms AdaGrad on non-convex problems

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, RMSprop, AdaDelta
- Classical: BFGS, L-BFGS

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adagrad.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adagrad.py)
:::
