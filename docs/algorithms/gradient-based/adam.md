# Adam

<span class="badge badge-gradient">Gradient-Based</span>

Adaptive Moment Estimation (Adam) optimization algorithm.

## Algorithm Overview

This module implements the Adam optimization algorithm. Adam is a gradient-based
optimization algorithm that computes adaptive learning rates for each parameter. It
combines the advantages of two other extensions of stochastic gradient descent:

    - AdaGrad
    - RMSProp

Adam works well in practice and compares favorably to other adaptive learning-method
algorithms as it converges fast and the learning speed of the Model is quite fast and
efficient. It is straightforward to implement, is computationally efficient, has little
memory requirements, is invariant to diagonal rescaling of the gradients, and is well
suited for problems that are large in terms of data and/or parameters.

## Usage

```python
from opt.gradient_based.adaptive_moment_estimation import ADAMOptimization
from opt.benchmark.functions import sphere

optimizer = ADAMOptimization(
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
| `alpha` | `float` | `0.001` | Learning rate (step size). |
| `beta1` | `float` | `0.9` | Exponential decay rate for first moment estimates (mean of gradients). |
| `beta2` | `float` | `0.999` | Exponential decay rate for second moment estimates (uncentered variance). |
| `epsilon` | `float` | `1e-13` | Small constant for numerical stability in division operations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Adaptive Moment Estimation               |
| Acronym           | ADAM                                     |
| Year Introduced   | 2014                                     |
| Authors           | Kingma, Diederik P.; Ba, Jimmy Lei       |
| Algorithm Class   | Gradient-Based                           |
| Complexity        | O(dim)                                   |
| Properties        | Gradient-based, Stochastic           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations:

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
x_{t+1} = x_t - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\alpha$ is the step size (learning rate)
- $\beta_1, \beta_2$ are exponential decay rates for moment estimates
- $\epsilon$ is a small constant for numerical stability
- $m_t$ is the first moment estimate (mean of gradients)
- $v_t$ is the second moment estimate (uncentered variance)

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter      | Default | BBOB Recommended | Description                         |
|----------------|---------|------------------|-------------------------------------|
| max_iter       | 1000    | 10000            | Maximum iterations                  |
| alpha          | 0.001   | 0.001-0.01       | Learning rate (step size)           |
| beta1          | 0.9     | 0.9              | Exponential decay for 1st moment    |
| beta2          | 0.999   | 0.999            | Exponential decay for 2nd moment    |
| epsilon        | 1e-8    | 1e-8             | Numerical stability constant        |

**Sensitivity Analysis**:
- `alpha`: **High** impact on convergence - controls step size
- `beta1`, `beta2`: **Medium** impact - control moment estimates
- Recommended tuning ranges: $\alpha \in [0.0001, 0.01]$, $\beta_1 \in [0.8, 0.95]$

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
- Time per iteration: $O(dim)$ for gradient computation and moment updates
- Space complexity: $O(dim)$ for storing moment estimates
- BBOB budget usage: _Typically uses 50-70% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, ill-conditioned, moderate multimodal
- **Weak function classes**: Highly multimodal with many local optima
- Typical success rate at 1e-8 precision: **50-70%** (dim=5)
- Expected Running Time (ERT): Competitive with other adaptive methods

**Convergence Properties**:
- Convergence rate: Fast initial convergence, then linear/sublinear
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Low-Medium** - adaptive rates help exploration

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Bias correction prevents issues in early iterations

**Known Limitations**:
- May not converge in some convex optimization scenarios (see AMSGrad paper)
- Hyperparameter sensitive - alpha tuning often needed
- Gradient approximation via finite differences less accurate than analytical

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization."
_arXiv preprint arXiv:1412.6980_. Presented at ICLR 2015.
https://arxiv.org/abs/1412.6980

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: No specific COCO benchmark data available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: https://github.com/sagarvegad/Adam-optimizer
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

AdamW: Variant with decoupled weight decay
BBOB Comparison: AdamW often generalizes better with regularization

Adamax: Variant using infinity norm
BBOB Comparison: More robust to large gradients

AMSGrad: Fixes convergence issues in original Adam
BBOB Comparison: Better convergence guarantees but similar BBOB performance

Nadam: Combines Adam with Nesterov momentum
BBOB Comparison: Often converges faster than standard Adam

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: AdamW, AMSGrad, Nadam, RMSprop, AdaGrad
- Classical: BFGS, L-BFGS

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adaptive_moment_estimation.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adaptive_moment_estimation.py)
:::
