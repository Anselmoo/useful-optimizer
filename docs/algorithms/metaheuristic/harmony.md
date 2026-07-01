# Harmony Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Harmony Search (HS) optimization algorithm.

## Algorithm Overview

This module implements the Harmony Search optimization algorithm. Harmony Search is a
metaheuristic algorithm inspired by the improvisation process of musicians. It is
commonly used for solving optimization problems.

The HarmonySearch class is the main class that implements the algorithm. It takes an
objective function, lower and upper bounds of the search space, dimensionality of the
search space, and other optional parameters. The search method runs the optimization
process and returns the best solution found and its fitness value.

## Usage

```python
from opt.metaheuristic.harmony_search import HarmonySearch
from opt.benchmark.functions import sphere

optimizer = HarmonySearch(
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
| `population_size` | `int` | `100` | Harmony memory size. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `harmony_memory_accepting_rate` | `float` | `0.95` | Probability of selecting a value
        from harmony memory (HMCR). |
| `pitch_adjusting_rate` | `float` | `0.7` | Probability of adjusting a selected harmony
        (PAR). |
| `bandwidth` | `float` | `0.01` | Range for pitch adjustment. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Harmony Search                           |
| Acronym           | HS                                       |
| Year Introduced   | 2001                                     |
| Authors           | Geem, Zong Woo; Kim, Joong Hoon; Loganathan, G.V. |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equation (harmony improvisation):

$$
x_i^{new} = \begin{cases}
x_i^{HM} + bw \cdot U(-1, 1) & \text{if } r_1 < HMCR \text{ and } r_2 < PAR \\
x_i^{HM} & \text{if } r_1 < HMCR \text{ and } r_2 \geq PAR \\
x_i^{random} & \text{if } r_1 \geq HMCR
\end{cases}
$$

where:
- $x_i^{new}$ is the new harmony component at dimension $i$
- $x_i^{HM}$ is randomly selected from harmony memory
- $HMCR$ is the harmony memory considering rate (0.95)
- $PAR$ is the pitch adjustment rate (0.7)
- $bw$ is the bandwidth for pitch adjustment (0.01)
- $r_1, r_2$ are random numbers in $[0, 1]$
- $U(-1, 1)$ is uniform random in $[-1, 1]$

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Harmony memory size            |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| harmony_memory_accepting_rate | 0.95 | 0.90-0.99   | Prob. of using harmony memory  |
| pitch_adjusting_rate   | 0.7     | 0.1-0.9          | Prob. of pitch adjustment      |
| bandwidth              | 0.01    | 0.001-0.1        | Pitch adjustment range         |

**Sensitivity Analysis**:
- `harmony_memory_accepting_rate`: **High** impact on exploration/exploitation balance
- `pitch_adjusting_rate`: **Medium** impact on local search intensity
- `bandwidth`: **Medium** impact on step size
- Recommended tuning ranges: $HMCR \in [0.90, 0.99]$, $PAR \in [0.1, 0.9]$, $bw \in [0.001, 0.1]$

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
- BBOB budget usage: _Typically uses 60-80% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, weakly-structured problems
- **Weak function classes**: Highly separable, ill-conditioned functions
- Typical success rate at 1e-8 precision: **15-25%** (dim=5)
- Expected Running Time (ERT): Moderate; competitive on complex landscapes

**Convergence Properties**:
- Convergence rate: Sublinear
- Local vs Global: Balanced; HMCR and PAR control trade-off
- Premature convergence risk: **Medium** (depends on parameter tuning)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Bandwidth prevents extreme step sizes

**Known Limitations**:
- Performance sensitive to HMCR, PAR, and bandwidth parameter settings
- May converge slowly on high-dimensional problems (dim > 20)
- BBOB known issues: Less effective on ill-conditioned problems

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). "A New Heuristic
Optimization Algorithm: Harmony Search."
_Simulation_, 76(2), 60-68.
https://doi.org/10.1177/003754970107600201

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Various MATLAB implementations available
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

SimulatedAnnealing: Temperature-based metaheuristic with similar exploration strategy
BBOB Comparison: Both effective on multimodal problems; HS more parameter-dependent

GeneticAlgorithm: Population-based evolutionary algorithm
BBOB Comparison: GA generally faster on separable functions; HS better on rotated problems

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
View the implementation: [`harmony_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/harmony_search.py)
:::
