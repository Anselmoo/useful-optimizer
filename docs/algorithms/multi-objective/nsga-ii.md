# NSGA-II

<span class="badge badge-multi">Multi-Objective</span>

Non-dominated Sorting Genetic Algorithm II (NSGA-II) multi-objective optimizer.

## Algorithm Overview

This module implements the NSGA-II algorithm, one of the most popular and
highly-cited multi-objective evolutionary optimization algorithms.

NSGA-II uses fast non-dominated sorting and crowding distance assignment
to maintain a well-spread Pareto-optimal front while efficiently converging
to the true Pareto front.

## Usage

```python
from opt.multi_objective.nsga_ii import NSGAII
from opt.benchmark.functions import sphere

optimizer = NSGAII(
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
| `objectives` | `Sequence[Callable]` | Required | List of objective functions to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `max_iter` | `int` | `200` | Maximum number of generations. |
| `seed` | `int  \|  None` | `None` | **REQUIRED for BBOB compliance. |
| `population_size` | `int` | `100` | Number of individuals in population. |
| `crossover_prob` | `float` | `_CROSSOVER_PROBABILITY` | SBX crossover probability. |
| `mutation_prob` | `float  \|  None` | `None` | Polynomial mutation probability per dimension. |
| `tournament_size` | `int` | `_TOURNAMENT_SIZE` | Binary tournament selection size. |
| `eta_c` | `float` | `_SBX_DISTRIBUTION_INDEX` | SBX distribution index. |
| `eta_m` | `float` | `_POLYNOMIAL_MUTATION_INDEX` | Polynomial mutation distribution index. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Non-dominated Sorting Genetic Algorithm II|
| Acronym           | NSGA-II                                  |
| Year Introduced   | 2002                                     |
| Authors           | Deb, Kalyanmoy; Pratap, Amrit; Agarwal, Sameer; Meyarivan, T |
| Algorithm Class   | Multi-Objective                          |
| Complexity        | O(mN²) per generation                    |
| Properties        | Population-based, Derivative-free         |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

**Pareto Dominance**: Solution $\mathbf{x}_1$ dominates $\mathbf{x}_2$ if:

$$
\forall i: f_i(\mathbf{x}_1) \leq f_i(\mathbf{x}_2) \land
\exists j: f_j(\mathbf{x}_1) < f_j(\mathbf{x}_2)
$$

**Non-dominated Sorting**: Population sorted into fronts $F_1, F_2, ..., F_k$
where $F_1$ contains non-dominated solutions (Pareto front).

**Crowding Distance**: For solution $i$ in front $F$, on objective $m$:

$$
d_i^m = \frac{f_m^{i+1} - f_m^{i-1}}{f_m^{\max} - f_m^{\min}}
$$

Total crowding distance: $d_i = \sum_{m=1}^{M} d_i^m$

**Selection**: Binary tournament based on:
1. Pareto rank (lower is better)
2. Crowding distance (higher is better for diversity)

**Constraint handling**:
- **Boundary conditions**: Clamping to bounds after crossover/mutation
- **Feasibility enforcement**: SBX and polynomial mutation respect bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of individuals          |
| max_iter               | 200     | 10000            | Maximum generations            |
| crossover_prob         | 0.9     | 0.9              | SBX crossover probability      |
| mutation_prob          | 1/dim   | 1/dim            | Polynomial mutation probability|
| tournament_size        | 2       | 2                | Binary tournament size         |
| eta_c                  | 20      | 15-30            | SBX distribution index         |
| eta_m                  | 20      | 15-30            | Mutation distribution index    |

**Sensitivity Analysis**:
- `eta_c, eta_m`: **Medium** impact - controls offspring spread
- `population_size`: **High** impact - larger populations improve diversity
- Recommended tuning ranges: $\text{eta} \in [10, 30]$, $\text{pop} \in [50, 200]$

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
- **Hypervolume (HV)**: Volume dominated by Pareto front
- **Inverted Generational Distance (IGD)**: Distance to reference front
- **Spread**: Distribution uniformity metric
- **Epsilon Indicator**: Convergence quality measure

## Raises

ValueError: If search space is invalid or function evaluation fails.

## Notes

- Returns first Pareto front (rank 0) solutions
- Uses self.seed for all random number generation
- BBOB: Returns Pareto front after max_iter generations

**Computational Complexity**:
- Time per generation: $O(mN^2)$ where $m$ = objectives, $N$ = population
- Space complexity: $O(mN)$ for population and fitness storage
- BBOB budget usage: _Typically 60-80% of dim*10000 budget for convergence_

**BBOB Performance Characteristics** (Multi-Objective):
- **Best function classes**: Separable, low-dimensional (2-3 objectives)
- **Weak function classes**: Many-objective (>3), highly multimodal
- Typical Hypervolume: **85-95%** of reference front (bi-objective, dim=5)
- IGD competitive with MOEA/D on ZDT/DTLZ benchmarks

**Convergence Properties**:
- Convergence rate: Typically linear to Pareto front
- Diversity: Excellent via crowding distance mechanism
- Premature convergence risk: **Low** due to elitism and diversity preservation

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees identical Pareto fronts
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Pareto Front Characteristics**:
- **Non-dominated sorting**: Fast O(mN²) algorithm ensures accurate ranking
- **Crowding distance**: Maintains well-spread solutions along Pareto front
- **Elitism**: Combines parent and offspring, selects best N individuals
- **Diversity maintenance**: Boundary solutions get infinite crowding distance

**Implementation Details**:
- Parallelization: Not supported (sequential evaluation)
- Constraint handling: Clamping to bounds after SBX/polynomial mutation
- Numerical stability: Uses epsilon (1e-14) to prevent division by zero

**Known Limitations**:
- Performance degrades with >3 objectives (many-objective problems)
- Crowding distance less effective in high-dimensional objective spaces
- BBOB known issues: May struggle with disconnected Pareto fronts

**Version History**:
- v0.1.0: Initial implementation with BBOB multi-objective compliance

## References

[1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
"A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II."
_IEEE Transactions on Evolutionary Computation_, 6(2), 182-197.
https://doi.org/10.1109/4235.996017

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob-biobj/
- Multi-objective test suite: https://numbbo.github.io/coco-doc/bbob-biobj/functions/
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original NSGA-II: KanGAL Lab, IIT Kanpur
- This implementation: Based on [1] with BBOB multi-objective compliance

## See Also

MOEAD: Decomposition-based multi-objective algorithm
BBOB Comparison: Faster on many-objective problems, NSGA-II better
for 2-3 objectives with complex Pareto fronts

SPEA2: Strength Pareto Evolutionary Algorithm 2
BBOB Comparison: Similar performance, SPEA2 uses archive with
density-based truncation vs NSGA-II crowding distance

AbstractMultiObjectiveOptimizer: Base class for multi-objective optimizers
opt.benchmark.functions: BBOB-compatible multi-objective test functions

Related Multi-Objective Algorithms:
- Evolutionary: MOEAD, SPEA2
- Indicator-based: IBEA, SMS-EMOA
- Decomposition: MOEA/D, RVEA

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
View the implementation: [`nsga_ii.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/multi_objective/nsga_ii.py)
:::
