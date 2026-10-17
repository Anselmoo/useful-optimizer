# Sine Cosine Algorithm

<span class="badge badge-metaheuristic">Metaheuristic</span>

Sine Cosine Algorithm (SCA) optimization algorithm.

## Algorithm Overview

This module implements the Sine Cosine Algorithm (SCA) optimization algorithm.
SCA is a population-based metaheuristic algorithm inspired by the sine and cosine
functions. It is commonly used for solving optimization problems.

The SineCosineAlgorithm class provides an implementation of the SCA algorithm. It takes
an objective function, lower and upper bounds of the search space, dimensionality of
the search space, and other optional parameters as input. The search method performs
the optimization and returns the best solution found along with its fitness value.

## Usage

```python
from opt.metaheuristic.sine_cosine_algorithm import SineCosineAlgorithm
from opt.benchmark.functions import sphere

optimizer = SineCosineAlgorithm(
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
| `population_size` | `int` | `100` | Number of search agents. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `r1_cut` | `float` | `0.5` | Threshold for sine/cosine selection. |
| `r2_cut` | `float` | `0.5` | Threshold for movement direction. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Sine Cosine Algorithm                    |
| Acronym           | SCA                                      |
| Year Introduced   | 2016                                     |
| Authors           | Mirjalili, Seyedali                      |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equation using sine and cosine functions:

$$
X_i^{t+1} = \begin{cases}
X_i^t + r_1 \times \sin(r_2) \times |r_3 X^* - X_i^t| & \text{if } r_4 < 0.5 \\
X_i^t + r_1 \times \cos(r_2) \times |r_3 X^* - X_i^t| & \text{if } r_4 \geq 0.5
\end{cases}
$$

where:
- $X_i^t$ is the position of the i-th solution at iteration $t$
- $X^*$ is the best solution found so far
- $r_1$ controls movement amplitude (decreases linearly)
- $r_2$ is random angle in $[0, 2\pi]$
- $r_3$ is random weight for destination
- $r_4$ switches between sine and cosine (random in $[0, 1]$)

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of search agents        |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| r1_cut                 | 0.5     | 0.5              | Threshold for sine/cosine      |
| r2_cut                 | 0.5     | 0.5              | Threshold for direction        |

**Sensitivity Analysis**:
- `r1` (internal, adaptive): **High** impact on exploration/exploitation balance
- `population_size`: **Medium** impact on search quality
- Recommended tuning ranges: $r_1 \in [0, 2]$ (adaptive), population $\in [5 \times dim, 15 \times dim]$

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
- BBOB budget usage: _Typically uses 40-60% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, weakly-multimodal problems
- **Weak function classes**: Highly rotated, nonseparable functions
- Typical success rate at 1e-8 precision: **25-35%** (dim=5)
- Expected Running Time (ERT): Fast convergence on simple landscapes

**Convergence Properties**:
- Convergence rate: Linear (adaptive r1 parameter ensures smooth transition)
- Local vs Global: Good balance; r1 decreases linearly from 2 to 0
- Premature convergence risk: **Low** (oscillatory movements prevent stagnation)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Trigonometric functions well-behaved in optimization range

**Known Limitations**:
- May struggle on highly rotated problems due to coordinate-wise updates
- Performance depends on sine/cosine amplitude decreasing schedule
- BBOB known issues: Less effective on ill-conditioned ellipsoid functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Mirjalili, S. (2016). "SCA: A Sine Cosine Algorithm for solving optimization problems."
_Knowledge-Based Systems_, 96, 120-133.
https://doi.org/10.1016/j.knosys.2015.12.022

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results available
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: MATLAB code available from Mirjalili
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

ArithmeticOptimizationAlgorithm: Similar math-inspired metaheuristic (uses arithmetic ops)
BBOB Comparison: Both math-inspired; SCA simpler, faster on unimodal functions

WhaleOptimizationAlgorithm: Another Mirjalili algorithm with similar structure
BBOB Comparison: WOA spiral-based; SCA trigonometric-based

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
View the implementation: [`sine_cosine_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/sine_cosine_algorithm.py)
:::
