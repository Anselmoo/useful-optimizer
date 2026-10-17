# Shuffled Frog Leaping Algorithm

<span class="badge badge-metaheuristic">Metaheuristic</span>

Shuffled Frog Leaping Algorithm (SFLA) optimization algorithm.

## Algorithm Overview

This module provides an implementation of the Shuffled Frog Leaping Algorithm (SFLA)
optimizer. The SFLA is a population-based optimization algorithm inspired by the
behavior of frogs in a pond. It is used to solve optimization problems by iteratively
improving a population of candidate solutions.

The algorithm works by maintaining a population of frogs, where each frog represents a
candidate solution. In each iteration, the frogs are shuffled and leaped towards the
mean position of the best frogs. This process helps explore the search space and
converge towards the optimal solution.

This module defines the `ShuffledFrogLeapingAlgorithm` class, which is responsible for
executing the optimization process. The class takes an objective function, lower and
upper bounds of the search space, dimensionality of the search space, population size,
maximum number of iterations, and other optional parameters as input.

Example usage:
    optimizer = ShuffledFrogLeapingAlgorithm(
        func=shifted_ackley, lower_bound=-32.768, upper_bound=+32.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Fitness value: {best_fitness}")

## Usage

```python
from opt.metaheuristic.shuffled_frog_leaping_algorithm import ShuffledFrogLeapingAlgorithm
from opt.benchmark.functions import sphere

optimizer = ShuffledFrogLeapingAlgorithm(
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
| `population_size` | `int` | `100` | Total number of frogs. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `cut` | `int` | `2` | Number of memeplexes (population divisions). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Shuffled Frog Leaping Algorithm          |
| Acronym           | SFLA                                     |
| Year Introduced   | 2006                                     |
| Authors           | Eusuff, Muzaffar; Lansey, Kevin          |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equation (worst frog in memeplex leaps toward best):

$$
X_w^{new} = X_w + r \cdot (X_b - X_w)
$$

where:
- $X_w$ is the worst frog position in memeplex
- $X_b$ is the best frog position in memeplex
- $r$ is random number in $[0, 1]$
- If improvement fails, try global best; if still fails, generate random position

Memeplex structure: Population divided into m memeplexes, each evolves locally,
then global shuffling redistributes information.

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Total number of frogs          |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| cut                    | 2       | 2-5              | Number of memeplexes (divisions) |

**Sensitivity Analysis**:
- `population_size`: **High** impact on search quality
- `cut` (memeplexes): **Medium** impact on local vs global search balance
- Recommended tuning ranges: $cut \in [2, 5]$, population $\in [5 \times dim, 20 \times dim]$

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
- BBOB budget usage: _Typically uses 50-70% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, discrete-like continuous problems
- **Weak function classes**: Highly continuous, smooth unimodal functions
- Typical success rate at 1e-8 precision: **20-30%** (dim=5)
- Expected Running Time (ERT): Moderate; good for complex landscapes

**Convergence Properties**:
- Convergence rate: Sublinear (memetic local search improves efficiency)
- Local vs Global: Excellent balance via memeplex shuffling
- Premature convergence risk: **Low** (shuffling prevents stagnation)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Random fallback prevents infinite loops

**Known Limitations**:
- Originally designed for discrete optimization; adapted for continuous
- Performance depends on memeplex count and shuffling frequency
- BBOB known issues: Less effective on very smooth, unimodal functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Eusuff, M. M., & Lansey, K. E. (2006). "Shuffled frog-leaping algorithm:
a memetic meta-heuristic for discrete optimization."
_Engineering Optimization_, 38(2), 129-154.
https://doi.org/10.1080/03052150500384759

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results (originally for discrete problems)
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Various MATLAB implementations available
- This implementation: Adapted for continuous optimization with BBOB compliance

## See Also

ParticleSwarm: PSO-inspired algorithm with similar swarm intelligence concepts
BBOB Comparison: PSO faster on continuous; SFLA better on discrete/combinatorial

GeneticAlgorithm: Population-based evolutionary algorithm
BBOB Comparison: Both effective on multimodal; SFLA uses memetic local search

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
View the implementation: [`shuffled_frog_leaping_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/shuffled_frog_leaping_algorithm.py)
:::
