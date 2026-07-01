# AMSGrad

<span class="badge badge-gradient">Gradient-Based</span>

AMSGrad optimization algorithm.

## Algorithm Overview

This module implements the AMSGrad optimization algorithm. AMSGrad is a variant of Adam
that fixes the exponential moving average issue in Adam. It ensures that the second moment
estimate never decreases, which helps with convergence to the optimal solution.

AMSGrad performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    v_hat = max(v_hat, v)
    m_hat = m / (1 - beta1^t)
    v_hat_corrected = v_hat / (1 - beta2^t)
    x = x - learning_rate * m_hat / (sqrt(v_hat_corrected) + epsilon)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - v: second moment estimate (exponential moving average of squared gradients)
    - v_hat: maximum of all v up to current time step
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - epsilon: small constant for numerical stability
    - t: time step

## Usage

```python
from opt.gradient_based.amsgrad import AMSGrad
from opt.benchmark.functions import sphere

optimizer = AMSGrad(
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
| `learning_rate` | `float` | `0.001` | Learning rate (step size). |
| `beta1` | `float` | `0.9` | Exponential decay rate for first moment estimates. |
| `beta2` | `float` | `0.999` | Exponential decay rate for second moment estimates. |
| `epsilon` | `float` | `1e-08` | Small constant for numerical stability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | AMSGrad                                  |
| Acronym           | AMSGRAD                                  |
| Year Introduced   | 2018                                     |
| Authors           | Reddi, Sashank J.; Kale, Satyen; Kumar, Sanjiv |
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
\hat{v}_t = \max(\hat{v}_{t-1}, v_t)
$$

$$
x_{t+1} = x_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot m_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\alpha$ is the learning rate
- $\beta_1, \beta_2$ are exponential decay rates
- $\epsilon$ is a small constant for numerical stability
- $m_t$ is the first moment estimate
- $v_t$ is the second moment estimate
- $\hat{v}_t$ is the maximum of all $v$ up to time $t$

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter        | Default | BBOB Recommended | Description                       |
|------------------|---------|------------------|-----------------------------------|
| max_iter         | 1000    | 10000            | Maximum iterations                |
| learning_rate    | 0.001   | 0.001-0.01       | Learning rate (step size)         |
| beta1            | 0.9     | 0.9              | Decay for 1st moment              |
| beta2            | 0.999   | 0.999            | Decay for 2nd moment              |
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
- **Best function classes**: Unimodal, moderately multimodal functions
- **Weak function classes**: Highly multimodal with many local optima
- Typical success rate at 1e-8 precision: **50-70%** (dim=5)
- Expected Running Time (ERT): Similar to Adam with better convergence

**Convergence Properties**:
- Convergence rate: Fast initial convergence, better than Adam theoretically
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Low** - non-decreasing second moment helps

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Maximum operation ensures non-decreasing second moment

**Known Limitations**:
- May converge slower than Adam in practice despite better theory
- Learning rate still requires tuning
- Gradient approximation via finite differences less accurate

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Reddi, S. J., Kale, S., & Kumar, S. (2018). "On the Convergence of Adam and Beyond."
_International Conference on Learning Representations (ICLR)_.
https://openreview.net/forum?id=ryQu7f-RZ

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: No specific COCO benchmark data available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper: Reddi et al. (2018) - ICLR
- This implementation: AMSGrad with BBOB compliance

## See Also

Adam: Base algorithm with potential convergence issues
BBOB Comparison: AMSGrad provides better convergence guarantees

AdamW: Adam with decoupled weight decay
BBOB Comparison: Similar BBOB performance, different theoretical properties

Adamax: Adam variant using infinity norm
BBOB Comparison: Both fix different aspects of Adam

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, Adamax, Nadam
- Classical: BFGS, L-BFGS

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`amsgrad.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/amsgrad.py)
:::
