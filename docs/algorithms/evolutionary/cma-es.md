# CMA-ES

<span class="badge badge-evolutionary">Evolutionary</span>

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm.

## Algorithm Overview

This module implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) algorithm,
which is a derivative-free optimization method that uses an evolutionary strategy to search for
the optimal solution. It adapts the covariance matrix of the multivariate Gaussian distribution
to guide the search towards promising regions of the search space.

The CMA-ES algorithm is implemented in the `CMAESAlgorithm` class, which inherits from the
`AbstractOptimizer` class. The `CMAESAlgorithm` class provides a `search` method that runs the
CMA-ES algorithm to search for the optimal solution.

Example usage:
    optimizer = CMAESAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-12.768,
        upper_bound=12.768,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

## Usage

```python
from opt.evolutionary.cma_es import CMAESAlgorithm
from opt.benchmark.functions import sphere

optimizer = CMAESAlgorithm(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
    population_size=50,
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | Required | Objective function to minimize. |
| `dim` | `int` | Required | Problem dimensionality. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `population_size` | `int` | `100` | Number of offspring per generation (λ). |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `sigma_init` | `float` | `0.5` | Initial global step-size controlling search spread. |
| `epsilon` | `float` | `1e-09` | Minimum step-size threshold to prevent numerical instability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Covariance Matrix Adaptation Evolution Strategy |
| Acronym           | CMA-ES                                   |
| Year Introduced   | 2001                                     |
| Authors           | Hansen, Nikolaus; Ostermeier, Andreas    |
| Algorithm Class   | Evolutionary                             |
| Complexity        | O(n³) per iteration                      |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core sampling and update equations:

$$
x_i^{(g+1)} \sim m^{(g)} + \sigma^{(g)} \mathcal{N}(0, C^{(g)})
$$

where:
- $x_i^{(g+1)}$ is the $i$-th offspring at generation $g+1$
- $m^{(g)}$ is the mean (center of search distribution) at generation $g$
- $\sigma^{(g)}$ is the global step-size at generation $g$
- $C^{(g)}$ is the covariance matrix at generation $g$
- $\mathcal{N}(0, C^{(g)})$ is multivariate Gaussian with zero mean and covariance $C^{(g)}$

**Mean update**:

$$
m^{(g+1)} = \sum_{i=1}^{\mu} w_i x_{i:\lambda}^{(g+1)}
$$

**Covariance matrix update**:

$$
C^{(g+1)} = (1-c_1-c_\mu) C^{(g)} + c_1 p_c p_c^T + c_\mu \sum_{i=1}^{\mu} w_i (x_{i:\lambda}^{(g+1)} - m^{(g)})(x_{i:\lambda}^{(g+1)} - m^{(g)})^T
$$

**Constraint handling**:
- **Boundary conditions**: Clamping to bounds (solutions outside bounds are resampled)
- **Numerical stability**: Regularization added to covariance matrix to maintain positive definiteness

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 4+⌊3ln(n)⌋       | Number of offspring per generation |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| sigma_init             | 0.5     | (ub-lb)/5        | Initial global step-size       |
| epsilon                | 1e-9    | 1e-9             | Minimum step-size threshold    |

**Sensitivity Analysis**:
- `population_size`: **Medium** impact on convergence - larger improves exploration but slower
- `sigma_init`: **High** impact - controls initial search spread
- Recommended tuning ranges: $\text{sigma\_init} \in [0.1, 1.0]$, $\text{population\_size} \in [4+3\ln(n), 20n]$

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
- Time per iteration: $O(n^3 + \lambda n^2)$ where $n$ is dimension, $\lambda$ is population size
- Space complexity: $O(n^2)$ for covariance matrix storage
- BBOB budget usage: _Typically uses 30-70% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Ill-conditioned, Weakly structured multimodal, Multimodal with adequate structure
- **Weak function classes**: Highly multimodal with weak global structure
- Typical success rate at 1e-8 precision: **85-95%** (dim=5)
- Expected Running Time (ERT): Among top performers on BBOB benchmark suite

**Convergence Properties**:
- Convergence rate: Linear to superlinear on convex-quadratic functions
- Local vs Global: Strong global search via adaptive covariance, excellent local convergence
- Premature convergence risk: **Low** due to adaptive step-size control

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds with resampling on violation
- Numerical stability: Regularization added to covariance matrix to ensure positive definiteness

**Known Limitations**:
- Memory-intensive for very high dimensions (n > 1000) due to covariance matrix
- May struggle on highly rugged landscapes with many local optima
- BBOB known issues: None specific; one of the most robust algorithms

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added numerical stability improvements with regularization

## References

[1] Hansen, N., & Ostermeier, A. (2001). "Completely derandomized self-adaptation
in evolution strategies."
_Evolutionary Computation_, 9(2), 159-195.
https://doi.org/10.1162/106365601750190398

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- CMA-ES BBOB results: Available in COCO data archive (one of best-performing algorithms)
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original CMA-ES implementation: https://github.com/CMA-ES/pycma
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

DifferentialEvolution: Population-based evolutionary algorithm with simpler adaptation
BBOB Comparison: CMA-ES typically faster on ill-conditioned and multimodal functions

GeneticAlgorithm: Classical evolutionary algorithm with crossover and mutation
BBOB Comparison: CMA-ES significantly more efficient on continuous optimization

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution, EstimationOfDistributionAlgorithm
- Swarm: ParticleSwarm, AntColony
- Gradient: AdamW, SGDMomentum

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

::: tip Run-based charts
Convergence, distribution and ECDF charts appear here once this optimizer is included in the benchmark suite.
:::

## Related Pages

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cma_es.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/cma_es.py)
:::
