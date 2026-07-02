# Firefly Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Firefly Algorithm (FA) optimization algorithm.

## Algorithm Overview

This module provides an implementation of the Firefly Algorithm optimization algorithm.
The Firefly Algorithm is a metaheuristic optimization algorithm inspired by the
flashing behavior of fireflies. It is commonly used to solve optimization problems by
simulating the behavior of fireflies in attracting each other.

The algorithm works by representing potential solutions as fireflies in a search space.
Each firefly's brightness is determined by its fitness value, with brighter fireflies
representing better solutions. Fireflies move towards brighter fireflies in the search
space, and their movements are influenced by attractiveness and light absorption
coefficients.

This implementation provides a class called FireflyAlgorithm, which can be used to
perform optimization using the Firefly Algorithm. The class takes an objective
function, lower and upper bounds of the search space, dimensionality of the search
space, and other optional parameters. The search method of the class runs the
Firefly Algorithm optimization and returns the best solution found.

Example usage:
    optimizer = FireflyAlgorithm(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=32.768,
        population_size=100,
        max_iter=1000,
        alpha=0.5,
        beta_0=1,
        gamma=1,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

## Usage

```python
from opt.swarm_intelligence.firefly_algorithm import FireflyAlgorithm
from opt.benchmark.functions import sphere

optimizer = FireflyAlgorithm(
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
| `population_size` | `int` | `100` | Number of fireflies in the population. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `alpha` | `float` | `0.5` | Randomization parameter controlling step size of random movement. |
| `beta_0` | `float` | `1` | Attractiveness coefficient at distance r=0. |
| `gamma` | `float` | `1` | Light absorption coefficient. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Firefly Algorithm                        |
| Acronym           | FA                                       |
| Year Introduced   | 2009                                     |
| Authors           | Yang, Xin-She                            |
| Algorithm Class   | Swarm Intelligence                       |
| Complexity        | O(population_size^2 * dim * max_iter)    |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations based on bioluminescent attraction:

$$
\beta(r) = \beta_0 e^{-\gamma r^2}
$$

$$
x_i^{t+1} = x_i^t + \beta_0 e^{-\gamma r_{ij}^2}(x_j^t - x_i^t) + \alpha \epsilon_i^t
$$

where:
- $x_i^t$ is the position of firefly $i$ at iteration $t$
- $r_{ij}$ is the Euclidean distance between fireflies $i$ and $j$
- $\beta_0$ is the attractiveness at distance $r = 0$
- $\gamma$ is the light absorption coefficient
- $\alpha$ governs the random movement step size
- $\epsilon_i^t \in [-1, 1]$ is a random vector

Brightness and attractiveness:
- Brightness: $I_i = f(x_i)$ (objective function value)
- Less bright fireflies move toward brighter ones
- Attractiveness decreases with distance due to light absorption

Constraint handling:
- **Boundary conditions**: Clamping to [lower_bound, upper_bound]
- **Feasibility enforcement**: Direct bound checking after each update

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of fireflies            |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| alpha                  | 0.5     | 0.2-0.8          | Randomization parameter        |
| beta_0                 | 1.0     | 0.8-1.2          | Attractiveness at r=0          |
| gamma                  | 1.0     | 0.01-100         | Light absorption coefficient   |

**Sensitivity Analysis**:
- `alpha`: **High** impact on exploration - controls randomness
- `gamma`: **High** impact on convergence - controls interaction distance
- `beta_0`: **Medium** impact - scales attraction strength
- Recommended tuning ranges: $\alpha \in [0.2, 0.8]$, $\gamma \in [0.01, 100]$

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
- Time per iteration: $O(\text{population\_size}^2 \times \text{dim})$
- Space complexity: $O(\text{population\_size} \times \text{dim})$
- BBOB budget usage: _Typically uses 70-90% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, Separable functions
- **Weak function classes**: Ill-conditioned, High-dimensional problems
- Typical success rate at 1e-8 precision: **30-40%** (dim=5)
- Expected Running Time (ERT): Competitive on multimodal, slower on unimodal

**Convergence Properties**:
- Convergence rate: Linear to sub-linear depending on gamma setting
- Local vs Global: Excellent for multimodal due to multiple attractors
- Premature convergence risk: **Low** - good diversity maintenance

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds after position updates
- Numerical stability: Uses NumPy operations for numerical stability

**Known Limitations**:
- Quadratic complexity can be slow for large populations
- Parameter gamma requires problem-specific tuning
- BBOB known issues: May struggle on high-dimensional ill-conditioned functions

**Version History**:
- v0.1.0: Initial implementation
- Current: BBOB-compliant with seed parameter support

## References

[1] Yang, X.-S. (2009). "Firefly Algorithms for Multimodal Optimization."
In: _Stochastic Algorithms: Foundations and Applications (SAGA 2009)_,
Lecture Notes in Computer Science, vol. 5792, Springer, pp. 169-178.
https://doi.org/10.1007/978-3-642-04944-6_14

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: https://arxiv.org/abs/1003.1466 (arXiv preprint)
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper: https://link.springer.com/chapter/10.1007/978-3-642-04944-6_14
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

BatAlgorithm: Another nature-inspired algorithm by Yang using echolocation
BBOB Comparison: BA and FA have similar performance on multimodal problems

ParticleSwarm: Classic swarm intelligence algorithm
BBOB Comparison: FA often shows better diversity maintenance

GlowwormSwarmOptimization: Similar light-based attraction mechanism
BBOB Comparison: FA generally more widely studied and benchmarked

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
- Swarm: ParticleSwarm, AntColony, BatAlgorithm
- Gradient: AdamW, SGDMomentum

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

Convergence, final-fitness distribution and performance profile on `rastrigin` (5D), averaged over independent runs (compared against representative baselines):

<ClientOnly>
  <BenchmarkCharts
    algorithm="FireflyAlgorithm"
    functionName="rastrigin"
    :dimension="5"
    :compareWith="['GreyWolfOptimizer', 'ParticleSwarm', 'AntColony']"
  />
</ClientOnly>

## Related Pages

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`firefly_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/firefly_algorithm.py)
:::
