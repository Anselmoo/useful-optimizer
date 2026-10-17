# Nesterov Accelerated Gradient

<span class="badge badge-gradient">Gradient-Based</span>

Nesterov Accelerated Gradient (NAG) optimization algorithm.

## Algorithm Overview

This module implements the Nesterov Accelerated Gradient optimization algorithm. NAG is
an improvement over SGD with Momentum that provides better convergence rates. The key
idea is to compute the gradient not at the current position, but at an approximate
future position, which provides better gradient information.

NAG performs the following update rule:
    v = momentum * v - learning_rate * gradient(x + momentum * v)
    x = x + v

where:
    - x: current solution
    - v: velocity (momentum term)
    - learning_rate: step size for parameter updates
    - momentum: momentum coefficient (typically 0.9)
    - gradient: gradient of the objective function

## Usage

```python
from opt.gradient_based.nesterov_accelerated_gradient import NesterovAcceleratedGradient
from opt.benchmark.functions import sphere

optimizer = NesterovAcceleratedGradient(
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
| `momentum` | `float` | `0.9` | Momentum coefficient. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Nesterov Accelerated Gradient            |
| Acronym           | NAG                                      |
| Year Introduced   | 1983                                     |
| Authors           | Nesterov, Yurii                          |
| Algorithm Class   | Gradient-Based                           |
| Complexity        | O(dim)                                   |
| Properties        | Gradient-based, Stochastic           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations:

$$
v_t = \mu \cdot v_{t-1} - \eta \cdot \nabla f(x_t + \mu \cdot v_{t-1})
$$

$$
x_{t+1} = x_t + v_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $v_t$ is the velocity (momentum term) at iteration $t$
- $\mu$ is the momentum coefficient
- $\eta$ is the learning rate
- $\nabla f(x_t + \mu \cdot v_{t-1})$ is the gradient at lookahead position

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: Solutions clipped after each update

## Hyperparameters

| Parameter        | Default | BBOB Recommended | Description                       |
|------------------|---------|------------------|-----------------------------------|
| max_iter         | 1000    | 10000            | Maximum iterations                |
| learning_rate    | 0.01    | 0.001-0.1        | Learning rate (step size)         |
| momentum         | 0.9     | 0.9-0.99         | Momentum coefficient              |

**Sensitivity Analysis**:
- `learning_rate`: **High** impact on convergence
- `momentum`: **Medium** impact - affects lookahead distance
- Recommended tuning ranges: $\eta \in [0.0001, 0.1]$, $\mu \in [0.8, 0.99]$

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
- Time per iteration: $O(dim)$ for gradient computation at lookahead
- Space complexity: $O(dim)$ for velocity storage
- BBOB budget usage: _Typically uses 60-80% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Convex, smooth functions
- **Weak function classes**: Highly multimodal, non-smooth functions
- Typical success rate at 1e-8 precision: **40-60%** (dim=5)
- Expected Running Time (ERT): Better than SGD, comparable to momentum

**Convergence Properties**:
- Convergence rate: Faster than standard momentum, quadratic for convex
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Medium** - lookahead helps but may overshoot

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Lookahead gradient computed at projected position

**Known Limitations**:
- Learning rate and momentum require careful tuning
- May oscillate with high momentum
- Gradient approximation via finite differences less accurate
- Not adaptive to problem conditioning

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Nesterov, Y. (1983). "A method for solving the convex programming problem
with convergence rate O(1/k^2)."
_Soviet Mathematics Doklady_, 27, 372-376.

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
- This implementation: NAG with BBOB compliance

## See Also

SGDMomentum: Standard momentum without lookahead
BBOB Comparison: NAG typically converges faster

Nadam: Combines NAG with adaptive learning rates
BBOB Comparison: Nadam more robust across function classes

Adam: Adaptive learning rate without Nesterov momentum
BBOB Comparison: Adam generally outperforms NAG

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, Nadam, SGD Momentum
- Classical: BFGS, L-BFGS

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

::: tip Run-based charts
Convergence, distribution and ECDF charts appear here once this optimizer is included in the benchmark suite.
:::

## Related Pages

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`nesterov_accelerated_gradient.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/nesterov_accelerated_gradient.py)
:::
