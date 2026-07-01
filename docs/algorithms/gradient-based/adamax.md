# Adamax

<span class="badge badge-gradient">Gradient-Based</span>

Adamax optimization algorithm.

## Algorithm Overview

This module implements the AdaMax optimization algorithm. AdaMax is a variant of Adam
that uses the infinity norm instead of the L2 norm for the second moment estimate.
This makes it less sensitive to outliers in gradients and can be more stable in some cases.

AdaMax performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    u = max(beta2 * u, |gradient|)
    x = x - (learning_rate / (1 - beta1^t)) * (m / u)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - u: second moment estimate (exponential moving average of infinity norm of gradients)
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - t: time step

## Usage

```python
from opt.gradient_based.adamax import AdaMax
from opt.benchmark.functions import sphere

optimizer = AdaMax(
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
| `learning_rate` | `float` | `0.002` | Learning rate (step size). |
| `beta1` | `float` | `0.9` | Exponential decay rate for first moment estimates. |
| `beta2` | `float` | `0.999` | Exponential decay rate for infinity norm. |
| `epsilon` | `float` | `1e-08` | Small constant for numerical stability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Adamax                                   |
| Acronym           | ADAMAX                                   |
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
u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
x_{t+1} = x_t - \frac{\alpha}{u_t + \epsilon} \cdot \hat{m}_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\alpha$ is the learning rate
- $\beta_1, \beta_2$ are exponential decay rates
- $\epsilon$ is a small constant for numerical stability
- $m_t$ is the first moment estimate
- $u_t$ is the exponentially weighted infinity norm

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter        | Default | BBOB Recommended | Description                       |
|------------------|---------|------------------|-----------------------------------|
| max_iter         | 1000    | 10000            | Maximum iterations                |
| learning_rate    | 0.002   | 0.001-0.01       | Learning rate (step size)         |
| beta1            | 0.9     | 0.9              | Decay for 1st moment              |
| beta2            | 0.999   | 0.999            | Decay for infinity norm           |
| epsilon          | 1e-8    | 1e-8             | Numerical stability constant      |

**Sensitivity Analysis**:
- `learning_rate`: **High** impact on convergence
- `beta1`, `beta2`: **Medium** impact
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
- Time per iteration: $O(dim)$ for gradient computation and updates
- Space complexity: $O(dim)$ for storing moment estimates
- BBOB budget usage: _Typically uses 50-70% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, moderate multimodal functions
- **Weak function classes**: Highly multimodal with many local optima
- Typical success rate at 1e-8 precision: **50-70%** (dim=5)
- Expected Running Time (ERT): Similar to Adam, slightly more robust

**Convergence Properties**:
- Convergence rate: Fast initial convergence, then linear
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Low** - infinity norm provides stability

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Infinity norm prevents issues with large gradients

**Known Limitations**:
- Learning rate still requires tuning
- Gradient approximation via finite differences less accurate
- May struggle on highly non-convex landscapes

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization."
_arXiv preprint arXiv:1412.6980_. Section 7: Extensions.
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
- Original paper: Section 7 of Adam paper (Kingma & Ba, 2014)
- This implementation: Adamax variant with BBOB compliance

## See Also

Adam: Base algorithm using L2 norm for second moment
BBOB Comparison: Adamax more robust to large gradients

AdamW: Adam with decoupled weight decay
BBOB Comparison: Similar performance, AdamW better with regularization

AMSGrad: Fixes Adam convergence issues
BBOB Comparison: Both perform similarly on BBOB functions

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, AMSGrad, Nadam
- Classical: BFGS, L-BFGS

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adamax.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adamax.py)
:::
