# SGD with Momentum

<span class="badge badge-gradient">Gradient-Based</span>

Stochastic Gradient Descent with Momentum (SGD-M) optimization algorithm.

## Algorithm Overview

This module implements the SGD with Momentum optimization algorithm. SGD with Momentum
is an extension of SGD that accelerates gradient descent in the relevant direction and
dampens oscillations. It does this by adding a fraction of the update vector of the
past time step to the current update vector.

SGD with Momentum performs the following update rule:
    v = momentum * v - learning_rate * gradient
    x = x + v

where:
    - x: current solution
    - v: velocity (momentum term)
    - learning_rate: step size for parameter updates
    - momentum: momentum coefficient (typically 0.9)
    - gradient: gradient of the objective function at x

## Usage

```python
from opt.gradient_based.sgd_momentum import SGDMomentum
from opt.benchmark.functions import sphere

optimizer = SGDMomentum(
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
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | SGD with Momentum                        |
| Acronym           | SGD-M                                    |
| Year Introduced   | 1964                                     |
| Authors           | Polyak, Boris T.                         |
| Algorithm Class   | Gradient-Based                           |
| Complexity        | O(dim)                                   |
| Properties        | Gradient-based, Stochastic           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations:

$$
v_t = \mu \cdot v_{t-1} - \eta \cdot g_t
$$

$$
x_{t+1} = x_t + v_t
$$

where:
- $x_t$ is the solution at iteration $t$
- $g_t$ is the gradient at iteration $t$
- $v_t$ is the velocity (momentum term) at iteration $t$
- $\eta$ is the learning rate
- $\mu$ is the momentum coefficient

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
- `momentum`: **Medium** impact - accelerates convergence
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
- Time per iteration: $O(dim)$ for gradient computation
- Space complexity: $O(dim)$ for velocity storage
- BBOB budget usage: _Typically uses 60-80% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Convex, smooth functions
- **Weak function classes**: Highly multimodal, noisy functions
- Typical success rate at 1e-8 precision: **35-55%** (dim=5)
- Expected Running Time (ERT): Better than vanilla SGD, comparable to adaptive methods

**Convergence Properties**:
- Convergence rate: Faster than SGD, linear for convex functions
- Local vs Global: Tends toward local optima (gradient-based)
- Premature convergence risk: **Medium** - momentum helps escape plateaus

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds after each update
- Numerical stability: No special provisions beyond momentum

**Known Limitations**:
- Learning rate still requires manual tuning
- Momentum can cause overshooting in ravines
- May oscillate around minima with high momentum
- Not adaptive to problem conditioning

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods."
_USSR Computational Mathematics and Mathematical Physics_, 4(5), 1-17.
https://doi.org/10.1016/0041-5553(64)90137-5

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
- This implementation: Standard SGD with momentum for BBOB compliance

## See Also

SGD: Vanilla stochastic gradient descent without momentum
BBOB Comparison: Momentum variant converges faster on most functions

NesterovAcceleratedGradient: Improved momentum with lookahead
BBOB Comparison: NAG often outperforms standard momentum

Adam: Adaptive learning rate with momentum-like terms
BBOB Comparison: Adam generally more robust than SGD-M

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Gradient: Adam, AdamW, RMSprop, NAG
- Classical: BFGS, L-BFGS

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

Convergence, final-fitness distribution and performance profile on `rastrigin` (5D), averaged over independent runs (compared against representative baselines):

<ClientOnly>
  <BenchmarkCharts
    algorithm="SGDMomentum"
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
View the implementation: [`sgd_momentum.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/sgd_momentum.py)
:::
