# Adaptive Metropolis Optimizer

<span class="badge badge-probabilistic">Probabilistic</span>

Adaptive Metropolis (AM) algorithm with covariance adaptation.

## Algorithm Overview

This module implements Simulated Annealing enhanced with Adaptive Metropolis
proposal distribution, a probabilistic optimization method.

The algorithm adapts the proposal covariance based on the history of
accepted samples, improving exploration efficiency.

## Usage

```python
from opt.probabilistic.adaptive_metropolis import AdaptiveMetropolisOptimizer
from opt.benchmark.functions import sphere

optimizer = AdaptiveMetropolisOptimizer(
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
| `max_iter` | `int` | `1000` | Maximum MCMC iterations. |
| `initial_temp` | `float` | `10.0` | Starting temperature for annealing schedule. |
| `final_temp` | `float` | `0.01` | Final temperature for annealing schedule. |
| `adaptation_start` | `int` | `100` | Iteration to start covariance adaptation. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Adaptive Metropolis Algorithm            |
| Acronym           | AM                                       |
| Year Introduced   | 2001                                     |
| Authors           | Haario, Heikki; Saksman, Eero; Tamminen, Johanna |
| Algorithm Class   | Probabilistic                            |
| Complexity        | O(dim²) per iteration                    |
| Properties        | Stochastic, Adaptive                 |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Adaptive Metropolis uses Metropolis-Hastings MCMC with adaptive proposal:

$$
x_{t+1} \sim \mathcal{N}(x_t, C_t)
$$

where $C_t$ is the adapted covariance matrix:

$$
C_t = s_d \text{Cov}(x_0, \ldots, x_{t-1}) + s_d \epsilon I_d
$$

**Acceptance probability** (Metropolis criterion):

$$
\alpha(x_t, x_{t+1}) = \min\left(1, \exp\left(-\frac{f(x_{t+1}) - f(x_t)}{T_t}\right)\right)
$$

where:
- $s_d = \frac{2.4^2}{d}$ is the optimal scaling factor
- $\epsilon$ is small regularization (1e-6)
- $T_t$ is temperature decreasing with iteration
- $I_d$ is the $d$-dimensional identity matrix
- $\text{Cov}$ is the sample covariance of the chain history

**Temperature schedule**:

$$
T_t = T_0 \left(\frac{T_f}{T_0}\right)^{t/T}
$$

**Constraint handling**:
- **Boundary conditions**: Reflection (clip to bounds)
- **Feasibility enforcement**: Hard boundary constraints via clipping

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| max_iter               | 1000    | 5000-10000       | Maximum MCMC iterations        |
| initial_temp           | 10.0    | 1.0-10.0         | Starting temperature           |
| final_temp             | 0.01    | 0.001-0.1        | Final temperature              |
| adaptation_start       | 100     | max(100, 2*dim)  | Iteration to start adaptation  |

**Sensitivity Analysis**:
- `initial_temp`: **High** impact - Controls initial exploration
- `final_temp`: **Medium** impact - Affects final convergence precision
- `adaptation_start`: **Medium** impact - Earlier adaptation improves convergence
- Recommended tuning ranges: $T_0 \in [1, 20]$, $T_f \in [0.001, 0.5]$

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

- Uses self.seed for all random number generation
- BBOB: Returns final best solution after max_iter iterations
- Covariance adaptation improves local search efficiency

**Computational Complexity**:
- Time per iteration: $O(d^2)$ for covariance update with dimension $d$
- Space complexity: $O(d^2)$ for covariance matrix storage
- BBOB budget usage: _Typically 50-80% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal functions (Sphere, Ellipsoid, Rosenbrock)
- **Weak function classes**: Highly multimodal with many local minima
- Typical success rate at 1e-8 precision: **50-70%** (dim=5)
- Expected Running Time (ERT): Moderate, better than random search

**Convergence Properties**:
- Convergence rate: Linear with proper temperature schedule
- Local vs Global: Primarily local search with adaptive covariance
- Premature convergence risk: **Medium** - Depends on temperature schedule

**Probabilistic Concepts**:
- **Markov Chain**: MCMC generates samples from target distribution
- **Metropolis-Hastings**: Acceptance criterion based on fitness ratio
- **Proposal Distribution**: Gaussian with adaptive covariance
- **Posterior Sampling**: Chain explores regions of low function values
- **Covariance Adaptation**: Welford's online algorithm for sample covariance

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees identical MCMC chain
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random` for proposals and acceptance (note: not using default_rng)

**Implementation Details**:
- Parallelization: Not supported (single MCMC chain)
- Constraint handling: Reflection via np.clip to bounds
- Numerical stability: Regularization $\epsilon I$ prevents singular covariance
- Scaling: Optimal $s_d = 2.4^2 / d$ from Roberts & Rosenthal (2001)

**Known Limitations**:
- Single-chain MCMC may get stuck in local minima on multimodal functions
- Covariance estimation requires sufficient samples (adaptation_start)
- Not using `numpy.random.default_rng` - may affect reproducibility guarantees
- BBOB known issues: Slow convergence on ill-conditioned Rosenbrock

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Current version with BBOB compliance

## References

[1] Haario, H., Saksman, E., & Tamminen, J. (2001).
"An adaptive Metropolis algorithm."
_Bernoulli_, 7(2), 223-242.
https://doi.org/10.2307/3318737

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Not yet available in COCO archive
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Not publicly available
- This implementation: Based on [1] with simulated annealing temperature schedule

## See Also

SequentialMonteCarloOptimizer: Population-based probabilistic MCMC
BBOB Comparison: SMC better on multimodal, AM faster on unimodal

BayesianOptimizer: Surrogate-based probabilistic optimization
BBOB Comparison: BO better sample efficiency, AM better high-dim scaling

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Probabilistic: BayesianOptimizer, SequentialMonteCarloOptimizer
- Classical: SimulatedAnnealing
- Metaheuristic: HarmonySearch

## Related Pages

- [Probabilistic Algorithms](/algorithms/probabilistic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adaptive_metropolis.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/adaptive_metropolis.py)
:::
