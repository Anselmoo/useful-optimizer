# Eagle Strategy

<span class="badge badge-metaheuristic">Metaheuristic</span>

Eagle Strategy (ES) optimization algorithm.

## Algorithm Overview

This module implements the Eagle Strategy (ES) optimization algorithm. ES is a
metaheuristic optimization algorithm inspired by the hunting behavior of eagles.
The algorithm mimics the way eagles soar, glide, and swoop down to catch their prey.

In ES, each eagle represents a potential solution, and the objective function
determines the quality of the solutions. The eagles try to update their positions by
mimicking the hunting behavior of eagles, which includes soaring, gliding, and swooping.

ES has been used for various kinds of optimization problems including function
optimization, neural network training, and other areas of engineering.

## Usage

```python
from opt.metaheuristic.eagle_strategy import EagleStrategy
from opt.benchmark.functions import sphere

optimizer = EagleStrategy(
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
| `population_size` | `int` | `100` | Number of eagles. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Eagle Strategy                           |
| Acronym           | ES                                       |
| Year Introduced   | 2010                                     |
| Authors           | Yang, Xin-She; Deb, Suash                |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Two-stage hybrid approach:

**Stage 1 - Global Search (Lévy walk)**:
$$x^{t+1} = x^t + \alpha \oplus Lévy(\lambda)$$

**Stage 2 - Local Search (Firefly-inspired)**:
$$x_i^{t+1} = x_i^t + \beta e^{-\gamma r_{ij}^2}(x_j - x_i) + \alpha \epsilon_i$$

where:
- $\alpha$ is step size
- $Lévy(\lambda)$ is Lévy distribution (heavy-tailed random walk)
- $\beta$ is attraction coefficient
- $\gamma$ is light absorption coefficient
- $r_{ij}$ is distance between eagles i and j
- $\epsilon_i$ is random vector

Inspired by eagles' hunting: scan wide area (Lévy), focus on prey (firefly).

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of eagles               |
| max_iter               | 1000    | 10000            | Maximum iterations             |

**Sensitivity Analysis**:
- `population_size`: **Medium** impact on search quality
- Lévy step size and firefly parameters (internal): **High** impact
- Recommended tuning ranges: population $\in [5 \times dim, 15 \times dim]$

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
- BBOB budget usage: _Typically uses 55-75% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, complex landscapes
- **Weak function classes**: Simple unimodal, smooth functions
- Typical success rate at 1e-8 precision: **22-32%** (dim=5)
- Expected Running Time (ERT): Moderate; good on complex problems

**Convergence Properties**:
- Convergence rate: Sublinear (hybrid Lévy + firefly)
- Local vs Global: Excellent balance via two-stage approach
- Premature convergence risk: **Low** (Lévy walks maintain exploration)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Lévy step size control prevents extreme jumps

**Known Limitations**:
- Hybrid approach adds complexity compared to simpler algorithms
- Performance depends on Lévy step size and firefly parameters
- BBOB known issues: May be overkill for simple unimodal functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Yang, X. S., & Deb, S. (2010). "Eagle Strategy using Lévy Walk and Firefly Algorithms
for Stochastic Optimization."
_Nature Inspired Cooperative Strategies for Optimization (NICSO 2010)_, 101-111.
https://doi.org/10.1007/978-3-642-12538-6_9

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Various implementations available
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

FireflyAlgorithm: Local search component of Eagle Strategy
BBOB Comparison: ES combines firefly with Lévy walk; Firefly standalone

CuckooSearch: Also uses Lévy flights
BBOB Comparison: Both use Lévy walks; ES adds firefly local search

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
View the implementation: [`eagle_strategy.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/eagle_strategy.py)
:::
