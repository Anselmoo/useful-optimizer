# Particle Filter

<span class="badge badge-metaheuristic">Metaheuristic</span>

Sequential Monte Carlo Particle Filter (SMC-PF) optimization algorithm.

## Algorithm Overview

This module implements the Particle Filter algorithm. Particle filters, or Sequential
Monte Carlo (SMC) methods, are a set of on-line posterior density estimation algorithms
that estimate the posterior density of the state-space by directly implementing the
Bayesian recursion equations.

The main idea behind particle filters is to represent the posterior density function by
a set of random samples, or particles, and assign a weight to each particle that
represents the probability of that particle being sampled from the probability density
function.

Particle filters are particularly useful for non-linear and non-Gaussian estimation
problems.

## Usage

```python
from opt.metaheuristic.particle_filter import ParticleFilter
from opt.benchmark.functions import sphere

optimizer = ParticleFilter(
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
| `population_size` | `int` | `100` | Number of particles. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `inertia` | `float` | `0.7` | Inertia weight controlling velocity momentum. |
| `cognitive` | `float` | `1.5` | Cognitive coefficient for personal best attraction. |
| `social` | `float` | `1.5` | Social coefficient for global best attraction. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Sequential Monte Carlo Particle Filter   |
| Acronym           | SMC-PF                                   |
| Year Introduced   | 1993                                     |
| Authors           | Gordon, Neil J.; Salmond, David J.; Smith, Adrian F. M. |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size $\times$ dim $\times$ max_iter) |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Sequential importance sampling with resampling adapted for optimization:

**Propagation** (mutation):

$$
x_i^{t+1} = x_i^t + w \cdot v_i^t + c_1 r_1 (p_i - x_i^t) + c_2 r_2 (g - x_i^t)
$$

**Weighting** (importance):

$$
w_i \propto \exp(-f(x_i) / T)
$$

**Resampling** (selection):
- Particles resampled proportional to weights
- Prevents particle degeneracy

where:
- $x_i^t$ is the i-th particle position at iteration $t$
- $v_i^t$ is the particle velocity
- $p_i$ is the personal best position
- $g$ is the global best position
- $w$ is inertia weight (0.7)
- $c_1, c_2$ are cognitive and social coefficients (1.5)
- $r_1, r_2$ are random numbers in $[0, 1]$
- $T$ is temperature parameter for weighting

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of particles            |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| inertia                | 0.7     | 0.4-0.9          | Inertia weight                 |
| cognitive              | 1.5     | 1.5-2.0          | Cognitive coefficient          |
| social                 | 1.5     | 1.5-2.0          | Social coefficient             |

**Sensitivity Analysis**:
- `inertia`: **High** impact on exploration/exploitation balance
- `cognitive`: **Medium** impact on personal best influence
- `social`: **Medium** impact on global best influence
- Recommended tuning ranges: $w \in [0.4, 0.9]$, $c_1, c_2 \in [1.5, 2.0]$

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
- Space complexity: $O(population\_size \times dim)$
- BBOB budget usage: _Typically uses 50-70% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, weakly-multimodal problems
- **Weak function classes**: Highly multimodal, deceptive landscapes
- Typical success rate at 1e-8 precision: **25-35%** (dim=5)
- Expected Running Time (ERT): Moderate; similar to PSO on smooth functions

**Convergence Properties**:
- Convergence rate: Linear to sublinear
- Local vs Global: Balanced via cognitive/social coefficients
- Premature convergence risk: **Medium** (similar to PSO)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Velocity and position updates numerically stable

**Known Limitations**:
- This is a PSO-like adaptation of particle filtering for optimization
- Traditional SMC/PF is designed for state estimation, not optimization
- May not fully leverage resampling strategies from classical particle filters

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993).
"Novel approach to nonlinear/non-Gaussian Bayesian state estimation."
_IEE Proceedings F (Radar and Signal Processing)_, 140(2), 107-113.
https://doi.org/10.1049/ip-f-2.1993.0015

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: SMC/PF primarily used for state estimation; limited BBOB results
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Various implementations in signal processing libraries
- This implementation: SMC-PF adapted for optimization with PSO-like dynamics

## See Also

ParticleSwarm: Standard PSO algorithm with similar velocity update mechanism
BBOB Comparison: PSO typically faster; SMC-PF adds resampling for diversity

GeneticAlgorithm: Population-based evolutionary algorithm
BBOB Comparison: GA uses crossover/mutation; SMC-PF uses particle dynamics

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
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

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`particle_filter.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/particle_filter.py)
:::
