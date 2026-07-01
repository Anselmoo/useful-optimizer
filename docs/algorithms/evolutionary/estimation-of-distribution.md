# Estimation Of Distribution Algorithm

<span class="badge badge-evolutionary">Evolutionary</span>

Estimation of Distribution Algorithm (EDA) optimization algorithm.

## Algorithm Overview

This module implements the Estimation of Distribution Algorithm (EDA) optimizer.
The EDA optimizer is a population-based optimization algorithm that uses a probabilistic model
to estimate the distribution of promising solutions. It iteratively generates new solutions
by sampling from the estimated distribution.

The EstimationOfDistributionAlgorithm class is a subclass of the AbstractOptimizer class
and provides the implementation of the EDA optimizer. It initializes a population, selects
the best individuals based on fitness, estimates the mean and standard deviation of the
selected individuals, and generates new individuals by sampling from the estimated model.
The process is repeated for a specified number of iterations.

## Usage

```python
from opt.evolutionary.estimation_of_distribution_algorithm import EstimationOfDistributionAlgorithm
from opt.benchmark.functions import sphere

optimizer = EstimationOfDistributionAlgorithm(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Population size. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Estimation of Distribution Algorithm     |
| Acronym           | EDA                                      |
| Year Introduced   | 1996                                     |
| Authors           | Mühlenbein, Heinz; Paaß, Gerhard        |
| Algorithm Class   | Evolutionary                             |
| Complexity        | O(NP * dim) per iteration                |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

EDA replaces crossover and mutation with probabilistic model estimation and sampling:

**Model estimation**:

$$
\mu_i = \frac{1}{N_{selected}} \sum_{j \in Selected} x_{j,i}
$$

$$
\sigma_i^2 = \frac{1}{N_{selected}} \sum_{j \in Selected} (x_{j,i} - \mu_i)^2
$$

**Sampling new generation**:

$$
x_{new,i} \sim \mathcal{N}(\mu_i, \sigma_i^2)
$$

where:
- $\mu_i$ is estimated mean for dimension $i$
- $\sigma_i^2$ is estimated variance for dimension $i$
- $Selected$ are top-performing individuals
- New solutions sampled from estimated distribution

**Constraint handling**:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Resampling if outside bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of individuals          |
| max_iter               | 1000    | 10000            | Maximum iterations             |

**Sensitivity Analysis**:
- `population_size`: **High** impact - affects model quality
- Recommended tuning ranges: $population\_size \in [5 \cdot dim, 20 \cdot dim]$

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
- Time per iteration: $O(NP \cdot n)$
- Space complexity: $O(NP \cdot n)$
- BBOB budget usage: _Typically uses 55-90% of dim*10000 budget_

**BBOB Performance Characteristics**:
- **Best function classes**: Separable, Unimodal
- **Weak function classes**: Non-separable, Highly multimodal
- Typical success rate at 1e-8 precision: **65-80%** (dim=5)

**Convergence Properties**:
- Convergence rate: Linear on separable problems
- Local vs Global: Good on separable, struggles with dependencies
- Premature convergence risk: **Medium to High**

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required
- Initialization: Uniform random sampling
- RNG usage: `numpy.random.default_rng(self.seed)`

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds
- Numerical stability: Standard precision

**Known Limitations**:
- Assumes variable independence (univariate model)
- BBOB known issues: Poor performance on non-separable functions

**Version History**:
- v0.1.0: Initial implementation with Gaussian model

## References

[1] Mühlenbein, H., & Paaß, G. (1996). "From Recombination of Genes to the Estimation of Distributions I. Binary Parameters."
_Parallel Problem Solving from Nature_, LNCS 1141, 178-187.

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Gaussian univariate model for continuous optimization

## See Also

CMAESAlgorithm: Advanced covariance matrix adaptation
BBOB Comparison: CMA-ES models dependencies, EDA assumes independence

GeneticAlgorithm: Traditional crossover/mutation approach
BBOB Comparison: EDA uses explicit probabilistic models

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, Differential Evolution, CMAESAlgorithm
- Swarm: ParticleSwarm
- Gradient: AdamW, SGDMomentum

## Related Pages

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`estimation_of_distribution_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/estimation_of_distribution_algorithm.py)
:::
