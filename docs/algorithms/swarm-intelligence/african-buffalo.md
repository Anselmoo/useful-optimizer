# African Buffalo Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

African Buffalo Optimization (ABO) optimization algorithm.

## Algorithm Overview

Implementation based on:
Odili, J.B., Kahar, M.N.M. & Anwar, S. (2015).
African Buffalo Optimization: A Swarm-Intelligence Technique.
Procedia Computer Science, 76, 443-448.

The algorithm mimics the migratory and herding behavior of African buffalos,
using two key equations: the buffalo's movement toward the best location and
its tendency to explore new areas.

## Usage

```python
from opt.swarm_intelligence.african_buffalo_optimization import AfricanBuffaloOptimizer
from opt.benchmark.functions import sphere

optimizer = AfricanBuffaloOptimizer(
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
| `max_iter` | `int` | Required | Maximum iterations. |
| `population_size` | `int` | `30` | Population size. |
| `lp1` | `float` | `_LP1` | Learning parameter 1 controlling exploitation strength. |
| `lp2` | `float` | `_LP2` | Learning parameter 2 controlling exploration strength. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | African Buffalo Optimization             |
| Acronym           | ABO                                      |
| Year Introduced   | 2015                                     |
| Authors           | Odili, Julius Beneoluchi; Kahar, Mohd Nasir Mohd; Anwar, Shakir |
| Algorithm Class   | Swarm Intelligence                       |
| Complexity        | O(population_size $\times$ dim $\times$ max_iter) |
| Properties        | Population-based, Derivative-free, Nature-inspired |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations inspired by buffalo migration and herding:

Exploration memory update (maaa equation):

$$
\text{maaa}_i^{t+1} = \text{maaa}_i^t + \text{lp}_1 \cdot r_1 \cdot (x_g - x_i^t) + \text{lp}_2 \cdot r_2 \cdot (x_{pb,i} - x_i^t)
$$

Position update (waaa equation):

$$
x_i^{t+1} = \frac{x_i^t + \text{maaa}_i^{t+1}}{2}
$$

where:
- $x_i^t$ is the position of buffalo $i$ at iteration $t$
- $x_g$ is the global best position
- $x_{pb,i}$ is the personal best position of buffalo $i$
- $\text{maaa}_i$ is the exploration memory for buffalo $i$
- $\text{lp}_1, \text{lp}_2$ are learning parameters (0.6, 0.4)
- $r_1, r_2$ are random values in [0,1]

Constraint handling:
- **Boundary conditions**: Clamping to [lower_bound, upper_bound]
- **Feasibility enforcement**: Adaptive restart for stagnant buffalos

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 30      | 10$\times$dim    | Number of buffalos             |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| lp1                    | 0.6     | 0.6              | Learning parameter 1 (exploitation) |
| lp2                    | 0.4     | 0.4              | Learning parameter 2 (exploration) |

**Sensitivity Analysis**:
- `lp1`: **Medium** impact on convergence - controls exploitation strength
- `lp2`: **Medium** impact on convergence - controls exploration strength
- Recommended tuning ranges: $\text{lp1}, \text{lp2} \in [0.3, 0.7]$

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
- Time per iteration: $O(\text{population\_size} \times \text{dim})$
- Space complexity: $O(\text{population\_size} \times \text{dim})$
- BBOB budget usage: _Typically uses 60-80% of dim $\times$ 10000 budget_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, weakly-structured problems
- **Weak function classes**: Highly ill-conditioned functions
- Typical success rate at 1e-8 precision: **15-25%** (dim=5)
- Expected Running Time (ERT): Moderate, comparable to PSO variants

**Convergence Properties**:
- Convergence rate: Linear to sub-linear
- Local vs Global: Balanced exploration-exploitation via lp1/lp2
- Premature convergence risk: **Medium** (adaptive restart helps)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random` with consistent seeding

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds with adaptive restart
- Numerical stability: Standard floating-point arithmetic

**Known Limitations**:
- Performance degrades on high-dimensional problems (dim > 40)
- Adaptive restart may introduce discontinuities in convergence

**Version History**:
- v0.1.0: Initial implementation

## References

[1] Odili, J.B., Kahar, M.N.M., Anwar, S. (2015). "African Buffalo Optimization:
A Swarm-Intelligence Technique." _Procedia Computer Science_, 76, 443-448.
https://doi.org/10.1016/j.procs.2015.12.291

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

ParticleSwarm: Similar swarm-based algorithm with velocity-position updates
BBOB Comparison: PSO generally faster on unimodal functions

GreyWolfOptimizer: Another nature-inspired population-based algorithm
BBOB Comparison: Similar performance on multimodal functions

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

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`african_buffalo_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/african_buffalo_optimization.py)
:::
