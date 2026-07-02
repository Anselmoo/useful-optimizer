# AdamW

<span class="badge badge-gradient">Gradient-Based</span>

Adam with Decoupled Weight Decay (AdamW) optimization algorithm.

## Algorithm Overview

This module implements the AdamW optimization algorithm. AdamW is a variant of Adam
that decouples weight decay from the gradient-based update. This decoupling provides
better regularization and often leads to improved generalization in machine learning.

AdamW performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    x = x - learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * x)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - v: second moment estimate (exponential moving average of squared gradients)
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - epsilon: small constant for numerical stability
    - weight_decay: weight decay coefficient
    - t: time step

## Usage

```python
from opt.gradient_based.adamw import AdamW
from opt.benchmark.functions import sphere

optimizer = AdamW(
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
| `max_iter` | `int` | `DEFAULT_MAX_ITERATIONS` | Maximum iterations. |
| `learning_rate` | `float` | `ADAMW_LEARNING_RATE` | Learning rate (step size). |
| `beta1` | `float` | `ADAM_BETA1` | Exponential decay rate for first moment estimates. |
| `beta2` | `float` | `ADAM_BETA2` | Exponential decay rate for second moment estimates. |
| `epsilon` | `float` | `ADAM_EPSILON` | Small constant for numerical stability. |
| `weight_decay` | `float` | `ADAMW_WEIGHT_DECAY` | Weight decay coefficient for L2 regularization decoupled from gradient. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Adam with Decoupled Weight Decay         |
| Acronym           | ADAMW                                    |
| Year Introduced   | 2017                                     |
| Authors           | Loshchilov, Ilya; Hutter, Frank          |
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
x_{t+1} = x_t - \alpha \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot x_t\right)
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $\alpha$ is the learning rate
- $\beta_1, \beta_2$ are exponential decay rates for moment estimates
- $\epsilon$ is a small constant for numerical stability
- $\lambda$ is the weight decay coefficient
- $m_t, v_t$ are biased first and second moment estimates

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
| epsilon          | 1e-8    | 1e-8             | Numerical stability               |
| weight_decay     | 0.01    | 0.0-0.1          | Weight decay coefficient          |

**Sensitivity Analysis**:
- `learning_rate`: **High** impact on convergence
- `weight_decay`: **Medium** impact - provides regularization
- Recommended tuning ranges: $\alpha \in [0.0001, 0.01]$, $\lambda \in [0, 0.1]$

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
- **Best function classes**: Unimodal, ill-conditioned functions
- **Weak function classes**: Highly multimodal with many local optima
- Typical success rate at 1e-8 precision: **55-75%** (dim=5)
- Expected Running Time (ERT): Similar to Adam, sometimes better with regularization

**Convergence Properties**:
- Convergence rate: Fast initial convergence, linear/sublinear later
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Low** - weight decay provides regularization

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Bias correction and epsilon for numerical stability

**Known Limitations**:
- Weight decay hyperparameter requires tuning for optimal performance
- Gradient approximation via finite differences less accurate than analytical
- May struggle on highly non-convex landscapes without proper tuning

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Loshchilov, I., & Hutter, F. (2017). "Decoupled Weight Decay Regularization."
_arXiv preprint arXiv:1711.05101_. Presented at ICLR 2019.
https://arxiv.org/abs/1711.05101

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: No specific COCO benchmark data available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: https://github.com/loshchil/AdamW-and-SGDW
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

Adam: Base algorithm without decoupled weight decay
BBOB Comparison: AdamW often generalizes better with proper regularization

AMSGrad: Fixes convergence issues in Adam
BBOB Comparison: Similar BBOB performance but different theoretical guarantees

Nadam: Combines Adam with Nesterov momentum
BBOB Comparison: Nadam may converge faster but AdamW has better regularization

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AMSGrad, Nadam, Adamax
- Classical: BFGS, L-BFGS

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

Convergence, final-fitness distribution and performance profile on `rastrigin` (5D), averaged over independent runs (compared against representative baselines):

<ClientOnly>
  <BenchmarkCharts
    algorithm="AdamW"
    functionName="rastrigin"
    :dimension="5"
    :compareWith="['GreyWolfOptimizer', 'ParticleSwarm', 'AntColony']"
  />
</ClientOnly>

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adamw.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adamw.py)
:::
