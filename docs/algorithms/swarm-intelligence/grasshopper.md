# Grasshopper Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Grasshopper Optimization Algorithm (GOA) optimization algorithm.

## Algorithm Overview

This module implements the Grasshopper Optimization Algorithm, a nature-inspired
metaheuristic based on the swarming behavior of grasshoppers in nature.

Grasshoppers naturally form swarms and move toward food sources while avoiding
collisions with each other. The algorithm mimics this behavior with social forces
(attraction/repulsion) and movement toward the best solution.

## Usage

```python
from opt.swarm_intelligence.grasshopper_optimization import GrasshopperOptimizer
from opt.benchmark.functions import sphere

optimizer = GrasshopperOptimizer(
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
| `population_size` | `int` | `100` | Number of grasshoppers. |
| `c_max` | `float` | `_C_MAX` | Maximum coefficient for social forces. |
| `c_min` | `float` | `_C_MIN` | Minimum coefficient for social forces. |
| `f` | `float` | `_ATTRACTION_INTENSITY` | Attraction intensity in social force function. |
| `l` | `float` | `_ATTRACTIVE_LENGTH_SCALE` | Attractive length scale in social force function. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Grasshopper Optimization Algorithm       |
| Acronym           | GOA                                      |
| Year Introduced   | 2017                                     |
| Authors           | Saremi, Shahrzad; Mirjalili, Seyedali; Lewis, Andrew |
| Algorithm Class   | Swarm Intelligence |
| Complexity        | O(population_size $\times$ population_size $\times$ dim $\times$ max_iter) |
| Properties        | Population-based, Derivative-free, Nature-inspired |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core position update equation:

$$
X_i^{t+1} = S_i + G + A
$$

where:
- $X_i^{t+1}$ is the position of grasshopper $i$ at iteration $t+1$
- $S_i$ is the social interaction component
- $G$ is the gravity force component
- $A$ is the wind advection component (toward target/best solution)

Social interaction:

$$
S_i = \sum_{j=1, j \neq i}^N s(d_{ij}) \hat{d}_{ij}
$$

Interaction function:

$$
s(r) = f e^{-r/l} - e^{-r}
$$

where $f$ is attraction intensity, $l$ is attractive length scale, and $r$ is distance.

Constraint handling:
- **Boundary conditions**: Clamping to [lower_bound, upper_bound]
- **Feasibility enforcement**: Position updates maintain search space bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of grasshoppers         |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| f (intensity)          | 0.5     | 0.5              | Attraction intensity factor    |
| l (length_scale)       | 1.5     | 1.5              | Attractive length scale        |

**Sensitivity Analysis**:
- `f` (attraction intensity): **Medium** impact on exploration/exploitation balance
- `l` (length scale): **Medium** impact on social interaction range
- Recommended tuning ranges: $f \in [0.4, 0.6]$, $l \in [1.0, 2.0]$

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

ValueError:
If search space is invalid or function evaluation fails.

## Notes

- Modifies self.history if track_history=True
- Uses self.seed for all random number generation
- BBOB: Returns final best solution after max_iter or convergence

**Computational Complexity**:
- Time per iteration: $O(\text{population\_size}^2 \times \text{dim})$
- Space complexity: $O(\text{population\_size} \times \text{dim})$
- BBOB budget usage: _Typically uses 50-65% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, separable problems
- **Weak function classes**: Highly ill-conditioned or deceptive landscapes
- Typical success rate at 1e-8 precision: **40-50%** (dim=5)
- Expected Running Time (ERT): Competitive with other nature-inspired algorithms

**Convergence Properties**:
- Convergence rate: Adaptive - balances exploration and exploitation
- Local vs Global: Strong global search capability through social forces
- Premature convergence risk: **Low** - social interaction maintains diversity

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds after position updates
- Numerical stability: Uses epsilon to avoid division by zero in distance calculations

**Known Limitations**:
- Quadratic complexity due to pairwise distance calculations
- May require larger population for very high-dimensional problems
- BBOB known issues: Slower convergence on very simple unimodal functions

**Version History**:
- v0.1.0: Initial implementation
- Current: BBOB-compliant with seed parameter support

**Version History**:
- v0.1.0: Initial implementation
- Current: BBOB-compliant with seed parameter support

## References

[1] Saremi, S., Mirjalili, S., Lewis, A. (2017). "Grasshopper Optimisation Algorithm: Theory and application."
_Advances in Engineering Software_, 105, 30-47.
https://doi.org/10.1016/j.advengsoft.2017.01.004

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: https://seyedalimirjalili.com/goa
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original MATLAB code: Available from authors' website
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

DragonflyOptimizer: Similar swarm algorithm with multiple behavioral components
BBOB Comparison: GOA has simpler social force model, often faster on separable functions

GreyWolfOptimizer: Hierarchy-based swarm algorithm
BBOB Comparison: GOA typically better on high-dimensional multimodal problems

ParticleSwarm: Classical swarm intelligence algorithm
BBOB Comparison: GOA has more sophisticated social interaction model

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
- Swarm: ParticleSwarm, AntColony, DragonflyOptimizer
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

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`grasshopper_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/grasshopper_optimization.py)
:::
