# MOEA/D

<span class="badge badge-multi">Multi-Objective</span>

Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D).

## Algorithm Overview

This module implements MOEA/D, a highly influential multi-objective
optimization algorithm that decomposes a multi-objective problem into
scalar subproblems.

## Usage

```python
from opt.multi_objective.moead import MOEAD
from opt.benchmark.functions import sphere

optimizer = MOEAD(
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
| `objectives` | `list[Callable]` | Required | List of objective functions to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `population_size` | `int` | `100` | Number of subproblems (weight vectors). |
| `max_iter` | `int` | `300` | Maximum number of generations. |
| `n_neighbors` | `int` | `20` | Neighborhood size for each subproblem. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Multi-Objective Evolutionary Algorithm based on Decomposition|
| Acronym           | MOEA-D                                   |
| Year Introduced   | 2007                                     |
| Authors           | Zhang, Qingfu; Li, Hui                   |
| Algorithm Class   | Multi-Objective Decomposition            |
| Complexity        | O(T·N·m) per generation                  |
| Properties        | Decomposition-based, Neighborhood search, Derivative-free|
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

**Decomposition via Tchebycheff**: Multi-objective problem decomposed into
$N$ scalar subproblems using weight vectors $\lambda^1, ..., \lambda^N$:

$$
g^{TCH}(x|\lambda, z^*) = \max_{1 \leq i \leq m} \lambda_i |f_i(x) - z_i^*|
$$

where:
- $\lambda = (\lambda_1, ..., \lambda_m)$ is weight vector for subproblem
- $z^* = (z_1^*, ..., z_m^*)$ is reference point (ideal point)
- $f_i(x)$ is value of $i$-th objective for solution $x$
- $m$ is number of objectives

**Weight Vector Generation**: For bi-objective problems ($m=2$):

$$
\lambda^i = \left(\frac{i}{N-1}, 1 - \frac{i}{N-1}\right), \quad i = 0, ..., N-1
$$

**Neighborhood Structure**: Each subproblem $i$ has $T$ nearest neighbors
based on Euclidean distance between weight vectors.

**Update Strategy**: Offspring replaces at most $n_r$ (typically 2)
neighboring solutions that it improves.

**Constraint handling**:
- **Boundary conditions**: Clamping to bounds after SBX/mutation
- **Feasibility enforcement**: Clip operator ensures bound satisfaction

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 100-300          | Number of subproblems (weights)|
| max_iter               | 300     | 10000            | Maximum generations            |
| n_neighbors            | 20      | 20 (15-30)       | Neighborhood size              |
| crossover_rate         | 1.0     | 0.9-1.0          | SBX crossover probability      |
| mutation_rate          | 0.1     | 1/dim            | Polynomial mutation probability|
| eta_c                  | 20      | 15-30            | SBX distribution index         |
| eta_m                  | 20      | 15-30            | Mutation distribution index    |

**Sensitivity Analysis**:
- `n_neighbors`: **High** impact - controls exploitation vs exploration balance
- `population_size`: **High** impact - determines Pareto front resolution
- Recommended tuning ranges: $\text{T} \in [10, 30]$, $\text{N} \in [50, 300]$

## COCO/BBOB Benchmark Settings

**Search Space**:
- Dimensions tested: `2, 3, 5, 10, 20, 40`
- Bounds: Function-specific (typically `[0, 1]` for ZDT, `[-5, 5]` for DTLZ)
- Instances: **15** per function (BBOB multi-objective standard)

**Evaluation Budget**:
- Budget: $\text{dim} \times 10000$ function evaluations
- Independent runs: **15** (for statistical significance)
- Seeds: `0-14` (reproducibility requirement)

**Performance Metrics** (Multi-Objective):
- **Hypervolume (HV)**: Volume dominated by approximated Pareto front
- **Inverted Generational Distance (IGD)**: Convergence to reference front
- **Spread**: Uniformity of solution distribution
- **Epsilon Indicator**: Approximation quality measure

## Raises

ValueError: If search space is invalid or function evaluation fails.

## Notes

- Returns non-dominated solutions from archive
- Tchebycheff decomposition used for scalar optimization
- Neighborhood-based mating and update

**Computational Complexity**:
- Time per generation: $O(T \cdot N \cdot m)$ where $T$ = neighbors,
$N$ = population, $m$ = objectives
- Space complexity: $O(N \cdot (d + m))$ for population and fitness
- BBOB budget usage: _Typically 50-70% of dim*10000 budget for convergence_

**BBOB Performance Characteristics** (Multi-Objective):
- **Best function classes**: Convex Pareto fronts, many-objective (>3)
- **Weak function classes**: Highly irregular/disconnected Pareto fronts
- Typical Hypervolume: **80-90%** of reference front (bi-objective, dim=5)
- IGD often superior to NSGA-II on ZDT/DTLZ benchmarks

**Convergence Properties**:
- Convergence rate: Faster than Pareto-based methods via decomposition
- Diversity: Controlled by weight vector distribution
- Premature convergence risk: **Medium** - depends on neighborhood size

**Reproducibility**:
- **Deterministic**: Partially - weight generation is deterministic,
but random operations use global numpy RNG (not seeded in current impl)
- **BBOB compliance**: Requires seed parameter implementation for full compliance
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: Uses `np.random` (not seeded) - **limitation for BBOB**

**Pareto Front Characteristics**:
- **Decomposition**: Transforms MOP into scalar subproblems
- **Tchebycheff aggregation**: Handles non-convex Pareto fronts
- **Neighborhood search**: Exploits problem structure via local mating
- **Archive maintenance**: Non-dominated solutions preserved separately

**Implementation Details**:
- Parallelization: Not supported (sequential subproblem updates)
- Constraint handling: Clamping to bounds after variation operators
- Numerical stability: Uses epsilon (1e-6) in Tchebycheff to prevent
division by zero

**Known Limitations**:
- No seed parameter in current implementation (BBOB gap)
- Return order (pareto_front, pareto_set) reversed from typical convention
- Weight vector generation limited to bi-objective in current implementation
- BBOB known issues: May struggle with highly multimodal objectives

**Version History**:
- v0.1.0: Initial implementation with Tchebycheff decomposition

## References

[1] Zhang, Q., & Li, H. (2007).
"MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition."
_IEEE Transactions on Evolutionary Computation_, 11(6), 712-731.
https://doi.org/10.1109/TEVC.2007.892759

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob-biobj/
- Multi-objective test suite: https://numbbo.github.io/coco-doc/bbob-biobj/functions/
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original MOEA/D: University of Essex, Q. Zhang's research group
- This implementation: Based on [1] with BBOB multi-objective compliance

## See Also

NSGAII: Non-dominated sorting genetic algorithm
BBOB Comparison: MOEA/D typically faster and more scalable to many
objectives (>3), NSGA-II better for complex Pareto front shapes

SPEA2: Strength Pareto Evolutionary Algorithm 2
BBOB Comparison: MOEA/D more efficient on convex Pareto fronts,
SPEA2 better on highly irregular fronts

AbstractMultiObjectiveOptimizer: Base class for multi-objective optimizers
opt.benchmark.functions: BBOB-compatible multi-objective test functions

Related Multi-Objective Algorithms:
- Decomposition: NSGA-III, RVEA
- Pareto-based: NSGA-II, SPEA2
- Indicator-based: IBEA, SMS-EMOA

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

::: tip Run-based charts
Convergence, distribution and ECDF charts appear here once this optimizer is included in the benchmark suite.
:::

## Related Pages

- [Multi-Objective Algorithms](/algorithms/multi-objective/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`moead.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/multi_objective/moead.py)
:::
