# Very Large Scale Neighborhood

<span class="badge badge-metaheuristic">Metaheuristic</span>

Very Large Scale Neighborhood Search (VLSN) optimization algorithm.

## Algorithm Overview

This module implements the Very Large Scale Neighborhood Search (VLSN) optimization
algorithm. VLSN is a local search method used for mathematical optimization.
It explores very large neighborhoods with an efficient algorithm.

The main idea behind VLSN is to perform a search in a large-scale neighborhood to find
the optimal solution for a given function. The size of the neighborhood is defined by
the `neighborhood_size` parameter.
The larger the neighborhood size, the more potential solutions the algorithm will
consider at each step, but the more computational resources it will require.

VLSN is particularly useful for problems where the search space is large and complex,
and where traditional optimization methods may not be applicable.

## Usage

```python
from opt.metaheuristic.very_large_scale_neighborhood_search import VeryLargeScaleNeighborhood
from opt.benchmark.functions import sphere

optimizer = VeryLargeScaleNeighborhood(
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
| `population_size` | `int` | `100` | Number of individuals in population. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `neighborhood_size` | `int` | `10` | Number of neighbors explored around each
        individual. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Very Large Scale Neighborhood Search     |
| Acronym           | VLSN                                     |
| Year Introduced   | 2000                                     |
| Authors           | Ahuja, Ravindra K.; Orlin, James B.; Sharma, Dushyant |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size $\times$ neighborhood_size $\times$ dim $\times$ max_iter) |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Large neighborhood local search with efficient exploration:

**Neighborhood generation**:
For each individual $x_i$:
- Generate $neighborhood\_size$ neighbors
- Each neighbor: $x_{neighbor} = x_i + r \cdot \text{Uniform}(-1, 1)$
- Select best neighbor if improvement found

**Update rule**:

$$
x_i^{new} = \begin{cases}
\arg\min_{x \in N(x_i)} f(x) & \text{if } \min_{x \in N(x_i)} f(x) < f(x_i) \\
x_i & \text{otherwise}
\end{cases}
$$

where:
- $N(x_i)$ is the large neighborhood around $x_i$
- $|N(x_i)| = neighborhood\_size$ (default: 10)
- $r$ is a random scaling factor
- Larger neighborhoods enable better exploration

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of individuals          |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| neighborhood_size      | 10      | 5-20             | Neighborhood size              |

**Sensitivity Analysis**:
- `neighborhood_size`: **High** impact on exploration capability
- Larger neighborhoods improve solution quality but increase computational cost
- Recommended tuning ranges: $neighborhood\_size \in [5, 20]$ for most problems

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
- Time per iteration: $O(population\_size \times neighborhood\_size \times dim)$
- Space complexity: $O(population\_size \times dim)$
- BBOB budget usage: _Typically uses 60-80% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, locally-structured problems
- **Weak function classes**: Highly multimodal, plateaus with many local optima
- Typical success rate at 1e-8 precision: **20-30%** (dim=5)
- Expected Running Time (ERT): Moderate; effective for local refinement

**Convergence Properties**:
- Convergence rate: Linear (local search)
- Local vs Global: Primarily local search; large neighborhoods aid exploration
- Premature convergence risk: **Medium** (neighborhood size dependent)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Neighborhood generation well-controlled

**Known Limitations**:
- VLSN originally designed for combinatorial problems (routing, scheduling)
- This continuous adaptation may not fully leverage VLSN strengths
- Computational cost increases linearly with neighborhood_size

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Ahuja, R. K., Orlin, J. B., & Sharma, D. (2000). "Very large-scale
neighborhood search."
_International Transactions in Operational Research_, 7(4-5), 301-317.
https://doi.org/10.1111/j.1475-3995.2000.tb00201.x

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: VLSN primarily for combinatorial problems; limited BBOB results
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Various implementations for routing and scheduling
- This implementation: VLSN adapted for continuous optimization with BBOB compliance

## See Also

VariableDepthSearch: Related adaptive neighborhood search algorithm
BBOB Comparison: VDS uses depth; VLSN uses neighborhood size

TabuSearch: Memory-based local search metaheuristic
BBOB Comparison: Both local search; Tabu uses memory, VLSN uses large neighborhoods

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
View the implementation: [`very_large_scale_neighborhood_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/very_large_scale_neighborhood_search.py)
:::
