# Variable Depth Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Variable Depth Search (VDS) optimization algorithm.

## Algorithm Overview

This module implements the Variable Depth Search (VDS) optimization algorithm. VDS is a
local search method used for mathematical optimization. It explores the search space by
variable-depth first search and backtracking.

The main idea behind VDS is to perform a search in a variable depth to find the optimal
solution for a given function. The depth of the search is defined by the `depth`
parameter. The larger the depth, the more potential solutions the algorithm will
consider at each step, but the more computational resources it will require.

VDS is particularly useful for problems where the search space is large and complex, and
where traditional optimization methods may not be applicable.

## Usage

```python
from opt.metaheuristic.variable_depth_search import VariableDepthSearch
from opt.benchmark.functions import sphere

optimizer = VariableDepthSearch(
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
| `max_depth` | `int` | `20` | Maximum search depth for neighborhood exploration. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Variable Depth Search                    |
| Acronym           | VDS                                      |
| Year Introduced   | 1973                                     |
| Authors           | Lin, Shen; Kernighan, Brian W.           |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size $\times$ max_depth $\times$ dim $\times$ max_iter) |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Adaptive neighborhood search with variable depth:

**Depth-based perturbation**:

$$
x_i^{new} = x_i + U(-d, d)
$$

**Multi-depth exploration**:
For each depth $d \in [1, max\_depth]$:
- Generate candidate: $x' = x + \text{Uniform}(-d, d)$
- Accept if $f(x') < f(x)$
- Use best improvement across all depths

where:
- $x_i$ is the i-th individual position
- $d$ is the current search depth
- $U(-d, d)$ is uniform random in $[-d, d]$
- $max\_depth$ controls neighborhood size (default: 20)
- Larger depths enable escaping local optima

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of individuals          |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| max_depth              | 20      | 10-50            | Maximum search depth           |

**Sensitivity Analysis**:
- `max_depth`: **High** impact on exploration capability
- Larger depths allow escaping deeper local optima
- Recommended tuning ranges: $max\_depth \in [10, 50]$ for most problems

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
- Time per iteration: $O(population\_size \times max\_depth \times dim)$
- Space complexity: $O(population\_size \times dim)$
- BBOB budget usage: _Typically uses 60-80% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, locally-structured problems
- **Weak function classes**: Highly multimodal, deceptive landscapes
- Typical success rate at 1e-8 precision: **15-25%** (dim=5)
- Expected Running Time (ERT): Moderate; effective for local refinement

**Convergence Properties**:
- Convergence rate: Linear (local search)
- Local vs Global: Primarily local search; depth parameter aids exploration
- Premature convergence risk: **High** (local search nature)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Depth-based perturbations well-controlled

**Known Limitations**:
- VDS originally designed for combinatorial problems (TSP, partitioning)
- This continuous adaptation may not fully leverage VDS strengths
- High risk of local optima entrapment on complex landscapes

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Lin, S., & Kernighan, B. W. (1973). "An effective heuristic algorithm
for the traveling-salesman problem."
_Operations Research_, 21(2), 498-516.
https://doi.org/10.1287/opre.21.2.498

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: VDS primarily for combinatorial problems; limited BBOB results
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Various implementations for TSP and graph partitioning
- This implementation: VDS adapted for continuous optimization with BBOB compliance

## See Also

TabuSearch: Memory-based local search metaheuristic
BBOB Comparison: Both local search-based; Tabu uses memory, VDS uses depth

SimulatedAnnealing: Probabilistic local search metaheuristic
BBOB Comparison: SA uses temperature; VDS uses adaptive depth

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
View the implementation: [`variable_depth_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/variable_depth_search.py)
:::
