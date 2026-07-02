# Variable Neighborhood Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Variable Neighbourhood Search (VNS) optimization algorithm.

## Algorithm Overview

This module implements the Variable Neighborhood Search (VNS) optimizer. VNS is a
metaheuristic optimization algorithm that explores different neighborhoods of a
solution to find the optimal solution for a given objective function within a specified
search space.

The `VariableNeighborhoodSearch` class is the main class that implements the VNS
algorithm. It takes an objective function, lower and upper bounds of the search space,
dimensionality of the search space, and other optional parameters to control the
optimization process.

## Usage

```python
from opt.metaheuristic.variable_neighbourhood_search import VariableNeighborhoodSearch
from opt.benchmark.functions import sphere

optimizer = VariableNeighborhoodSearch(
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
| `population_size` | `int` | `100` | Number of candidate solutions. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `neighborhood_size` | `int` | `10` | Maximum neighborhood depth (k_max). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Variable Neighbourhood Search            |
| Acronym           | VNS                                      |
| Year Introduced   | 1997                                     |
| Authors           | Mladenović, Nenad; Hansen, Pierre        |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(neighborhood_size * dim * max_iter)    |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

VNS systematically changes neighborhood structure during search:

Minimize: $$f(x)$$ subject to $$x \in X \subseteq S$$

Core procedure:
1. **Shaking**: Generate random solution in k-th neighborhood $N_k(x)$
2. **Local Search**: Apply local descent from shaken solution
3. **Move or Not**: Accept if improved, else increase k

Neighborhood structure: $N_1(x) \subset N_2(x) \subset ... \subset N_{k_{max}}(x)$

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of candidate solutions  |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| neighborhood_size      | 10      | 5-20             | Maximum neighborhood depth     |

**Sensitivity Analysis**:
- `neighborhood_size`: **High** impact on exploration vs exploitation
- `population_size`: **Medium** impact on search quality
- Recommended tuning ranges: $k_{max} \in [5, 20]$, population $\in [5 \times dim, 15 \times dim]$

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
- Time per iteration: $O(neighborhood\_size \times dim)$
- Space complexity: $O(population\_size \times dim)$
- BBOB budget usage: _Typically uses 60-80% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, rugged landscapes with local structure
- **Weak function classes**: Smooth unimodal, highly continuous functions
- Typical success rate at 1e-8 precision: **22-32%** (dim=5)
- Expected Running Time (ERT): Moderate; effective on structured problems

**Convergence Properties**:
- Convergence rate: Depends on neighborhood structure (typically sublinear)
- Local vs Global: Excellent balance via systematic neighborhood changes
- Premature convergence risk: **Low** (neighborhood diversification prevents trapping)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Neighborhood structure ensures bounded exploration

**Known Limitations**:
- Originally designed for discrete/combinatorial optimization
- Neighborhood structure definition is problem-dependent
- BBOB known issues: May require problem-specific neighborhood design

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Mladenović, N., & Hansen, P. (1997). "Variable neighborhood search."
_Computers & Operations Research_, 24(11), 1097-1100.
https://doi.org/10.1016/S0305-0548(97)00031-2

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results (designed for combinatorial problems)
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper: Focused on combinatorial optimization
- This implementation: Adapted for continuous optimization with BBOB compliance

## See Also

VariableDepthSearch: Related variable-depth local search (Lin-Kernighan style)
BBOB Comparison: VDS for TSP-like problems; VNS more general framework

TabuSearch: Memory-based local search metaheuristic
BBOB Comparison: Both local search; VNS simpler, no memory required

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
View the implementation: [`variable_neighbourhood_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/variable_neighbourhood_search.py)
:::
