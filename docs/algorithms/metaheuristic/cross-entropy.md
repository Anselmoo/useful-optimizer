# Cross Entropy Method

<span class="badge badge-metaheuristic">Metaheuristic</span>

Cross-Entropy Method (CEM) optimization algorithm.

## Algorithm Overview

This module provides an implementation of the Cross-Entropy Method (CEM) optimizer. The
CEM algorithm is a stochastic optimization method that is particularly effective for
solving problems with continuous search spaces.

The CrossEntropyMethod class is the main class of this module and serves as the
optimizer. It takes an objective function, lower and upper bounds of the search space,
dimensionality of the search space, and other optional parameters as input. It uses the
CEM algorithm to find the optimal solution for the given objective function within the
specified search space.

Example usage:
    optimizer = CrossEntropyMethod(
        func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=+2.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.metaheuristic.cross_entropy_method import CrossEntropyMethod
from opt.benchmark.functions import sphere

optimizer = CrossEntropyMethod(
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
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `population_size` | `int` | `100` | Number of samples per iteration. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `elite_frac` | `float` | `0.2` | Fraction of samples to use as elite set. |
| `noise_decay` | `float` | `0.99` | Covariance decay factor to maintain exploration. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Cross-Entropy Method                     |
| Acronym           | CEM                                      |
| Year Introduced   | 1999                                     |
| Authors           | Rubinstein, Reuven Y.; Kroese, Dirk P.  |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Iteratively updates probability distribution to concentrate on better solutions:

$$\theta_{t+1} = \arg\max_\theta \sum_{x \in \mathcal{E}_t} \log f(x; \theta)$$

where:
- $\theta$ are distribution parameters (mean, covariance for Gaussian)
- $\mathcal{E}_t$ is the elite set (top performing solutions)
- $f(x; \theta)$ is the sampling distribution

For continuous optimization (Gaussian):
- $\mu_{t+1} = \frac{1}{|\mathcal{E}|} \sum_{x \in \mathcal{E}} x$
- $\Sigma_{t+1} = \frac{1}{|\mathcal{E}|} \sum_{x \in \mathcal{E}} (x - \mu_{t+1})(x - \mu_{t+1})^T$

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of samples per iteration|
| max_iter               | 1000    | 10000            | Maximum iterations             |
| elite_frac             | 0.2     | 0.1-0.3          | Fraction of elite samples      |
| noise_decay            | 0.99    | 0.95-1.0         | Covariance decay factor        |

**Sensitivity Analysis**:
- `elite_frac`: **High** impact on convergence speed vs stability
- `noise_decay`: **Medium** impact on exploration maintenance
- Recommended tuning ranges: $elite\_frac \in [0.1, 0.3]$, $noise\_decay \in [0.95, 1.0]$

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
- Time per iteration: $O(population\_size \times dim)$
- Space complexity: $O(population\_size \times dim + dim^2)$ (covariance matrix)
- BBOB budget usage: _Typically uses 40-60% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, weakly-multimodal, smooth landscapes
- **Weak function classes**: Highly multimodal, plateaus with many local optima
- Typical success rate at 1e-8 precision: **30-40%** (dim=5)
- Expected Running Time (ERT): Fast on smooth functions; excellent convergence

**Convergence Properties**:
- Convergence rate: Linear to superlinear on smooth functions
- Local vs Global: Strong exploitation via distribution focusing
- Premature convergence risk: **Medium** (elite selection can cause early convergence)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Covariance decay prevents degeneracy

**Known Limitations**:
- Can converge prematurely if elite_frac too small
- Requires sufficient population size for accurate distribution estimation
- BBOB known issues: May struggle on highly multimodal functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Rubinstein, R. Y. (1999). "The Cross-Entropy Method for Combinatorial and
Continuous Optimization."
_Methodology and Computing in Applied Probability_, 1(2), 127-190.
https://doi.org/10.1023/A:1010091220143

[2] Rubinstein, R. Y., & Kroese, D. P. (2004). "The Cross-Entropy Method:
A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation
and Machine Learning." Springer.

[3] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Available in various languages
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

CovarianceMatrixAdaptation: CMA-ES uses similar distribution adaptation
BBOB Comparison: CMA-ES more sophisticated covariance updates; CEM simpler

EvolutionStrategy: ES family of algorithms
BBOB Comparison: Both distribution-based; ES more specialized

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
- Swarm: ParticleSwarm, AntColony
- Gradient: AdamW, SGDMomentum

## Related Pages

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cross_entropy_method.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/cross_entropy_method.py)
:::
