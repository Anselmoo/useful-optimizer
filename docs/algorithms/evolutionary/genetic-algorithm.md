# Genetic Algorithm

<span class="badge badge-evolutionary">Evolutionary</span>

Genetic Algorithm (GA) optimization algorithm.

## Algorithm Overview

This module implements a genetic algorithm (GA) optimizer. Genetic algorithms are a
part of evolutionary computing, which is a rapidly growing area of artificial
intelligence.

The GA optimizer starts with a population of candidate solutions to an optimization
problem and evolves this population by iteratively applying a set of genetic operators.

Key components of the GA optimizer include:
- Initialization: The population is initialized with a set of random solutions.
- Selection: Solutions are selected to reproduce based on their fitness. The better the
    solutions, the more chances they have to reproduce.
- Crossover (or recombination): Pairs of solutions are selected for reproduction to
    create one or more offspring, in which each offspring consists of a mix of the
    parents' traits.
- Mutation: After crossover, the offspring are mutated with a small probability.
    Mutation introduces small changes in the solutions, providing genetic diversity.
- Replacement: The population is updated to include the new, fitter solutions.

The GA optimizer is suitable for solving both constrained and unconstrained optimization
problems. It's particularly useful for problems where the search space is large and
complex, and where traditional optimization methods may not be applicable.

## Usage

```python
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm
from opt.benchmark.functions import sphere

optimizer = GeneticAlgorithm(
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
| `population_size` | `int` | `150` | Number of individuals. |
| `max_iter` | `int` | `1000` | Maximum iterations/generations. |
| `tournament_size` | `int` | `3` | Number of individuals in tournament selection. |
| `crossover_rate` | `float` | `0.7` | Probability of inheriting from first parent in crossover. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Genetic Algorithm                        |
| Acronym           | GA                                       |
| Year Introduced   | 1975                                     |
| Authors           | Holland, John H.                         |
| Algorithm Class   | Evolutionary                             |
| Complexity        | O(NP * dim) per iteration                |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core operations (selection, crossover, mutation):

**Selection** (Tournament):
- Select $k$ random individuals
- Choose best among them: $p_{selected} = \arg\min_{p \in T_k} f(p)$

**Crossover** (Uniform):

$$
c_i = \begin{cases}
p1_i & \text{if } \text{rand}(0,1) < CR \\
p2_i & \text{otherwise}
\end{cases}
$$

**Mutation** (Gaussian):

$$
x'_i = x_i + \mathcal{N}(0, \sigma^2) \cdot (ub - lb) \cdot \text{rand}(0,1)
$$

where:
- $p1, p2$ are parent individuals
- $c$ is offspring
- $CR$ is crossover rate
- $\sigma$ controls mutation strength
- $ub, lb$ are upper and lower bounds
- Tournament size $k$ controls selection pressure

**Constraint handling**:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Offspring clipped to valid range

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 150     | 10*dim - 20*dim  | Number of individuals          |
| max_iter               | 1000    | 10000            | Maximum iterations/generations |
| tournament_size        | 3       | 2-5              | Tournament selection size      |
| crossover_rate         | 0.7     | 0.6-0.9          | Crossover probability          |

**Sensitivity Analysis**:
- `tournament_size`: **Medium** impact - higher increases selection pressure
- `crossover_rate`: **Medium** impact - balance exploration/exploitation
- Recommended tuning ranges: $tournament\_size \in [2, 7]$, $crossover\_rate \in [0.5, 0.95]$

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
- Time per iteration: $O(NP \cdot n + NP \log NP)$ for tournament selection
- Space complexity: $O(NP \cdot n)$ for population storage
- BBOB budget usage: _Typically uses 60-95% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Separable, Moderately structured
- **Weak function classes**: Ill-conditioned, Highly multimodal (compared to modern variants)
- Typical success rate at 1e-8 precision: **50-70%** (dim=5)
- Expected Running Time (ERT): Moderate; foundational but outperformed by modern algorithms

**Convergence Properties**:
- Convergence rate: Sub-linear on most functions
- Local vs Global: Good exploration, moderate exploitation
- Premature convergence risk: **Medium** - depends on selection pressure and diversity

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Standard floating-point precision

**Known Limitations**:
- Less efficient than modern evolutionary algorithms (DE, CMA-ES) on continuous optimization
- Performance highly dependent on parameter tuning
- BBOB known issues: None specific; well-studied baseline algorithm

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Current BBOB-compliant version with real-valued encoding

## References

[1] Holland, J. H. (1975). "Adaptation in Natural and Artificial Systems."
_University of Michigan Press_, Ann Arbor.
(Republished by MIT Press, 1992)

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- GA results: Foundational algorithm with extensive BBOB testing
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Classic GA with tournament selection, uniform crossover, Gaussian mutation
- This implementation: Based on [1] with real-valued encoding for BBOB compliance

## See Also

DifferentialEvolution: Modern evolutionary algorithm often outperforming GA on continuous problems
BBOB Comparison: DE typically faster convergence on continuous optimization

CMAESAlgorithm: Covariance-based evolutionary strategy
BBOB Comparison: CMA-ES significantly more efficient on continuous problems

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: DifferentialEvolution, CMAESAlgorithm, EstimationOfDistributionAlgorithm
- Swarm: ParticleSwarm, AntColony
- Gradient: AdamW, SGDMomentum

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

Convergence, final-fitness distribution and performance profile on `rastrigin` (5D), averaged over independent runs (compared against representative baselines):

<ClientOnly>
  <BenchmarkCharts
    algorithm="GeneticAlgorithm"
    functionName="rastrigin"
    :dimension="5"
    :compareWith="['GreyWolfOptimizer', 'ParticleSwarm', 'AntColony']"
  />
</ClientOnly>

## Related Pages

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`genetic_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/genetic_algorithm.py)
:::
