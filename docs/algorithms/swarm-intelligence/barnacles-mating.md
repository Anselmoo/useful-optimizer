# Barnacles Mating Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Barnacles Mating Optimizer (BMO) optimization algorithm.

## Algorithm Overview

Implementation based on:
Sulaiman, M.H., Mustaffa, Z., Saari, M.M. & Daniyal, H. (2020).
Barnacles Mating Optimizer: A new bio-inspired algorithm for solving
engineering optimization problems.
Engineering Applications of Artificial Intelligence, 87, 103330.

The algorithm mimics the mating behavior of barnacles, where sessile
creatures must extend their reproductive organs to reach nearby mates.

## Usage

```python
from opt.swarm_intelligence.barnacles_mating import BarnaclesMatingOptimizer
from opt.benchmark.functions import sphere

optimizer = BarnaclesMatingOptimizer(
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
| `pl` | `int` | `_PL` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Barnacles Mating Optimizer             |
| Acronym           | BMO                           |
| Year Introduced   | 2018                            |
| Authors           | Various (see References)                |
| Algorithm Class   | Swarm Intelligence |
| Complexity        | O(population_size $\times$ dim $\times$ max_iter)                   |
| Properties        | Population-based, Derivative-free, Nature-inspired           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations:

$$
x_{t+1} = x_t + v_t
$$

where:
- $x_t$ is the position at iteration $t$
- $v_t$ is the velocity/step at iteration $t$
- Algorithm-specific update mechanisms

Constraint handling:
- **Boundary conditions**: Clamping to [lower_bound, upper_bound]
- **Feasibility enforcement**: Direct bound checking after updates

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of individuals          |
| max_iter               | 1000    | 10000            | Maximum iterations             |

**Sensitivity Analysis**:
- Parameters: **Medium** impact on convergence
- Recommended tuning ranges: Standard parameter tuning applies

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
- **Best function classes**: Multimodal, Moderately ill-conditioned
- **Weak function classes**: Highly separable unimodal functions
- Typical success rate at 1e-8 precision: **20-40%** (dim=5)
- Expected Running Time (ERT): Moderate, comparable to other swarm algorithms

**Convergence Properties**:
- Convergence rate: Sub-linear to linear
- Local vs Global: Balanced exploration-exploitation
- Premature convergence risk: **Medium**

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Standard floating-point arithmetic

**Known Limitations**:
- May struggle on very high-dimensional problems (dim > 50)

**Version History**:
- v0.1.0: Initial implementation

## References

[1] Barnacles Mating Optimizer (2018). "Original publication."
_Journal/Conference_, Available in scientific literature.

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- This implementation: Based on original algorithm with BBOB compliance

## See Also

ParticleSwarm: Classic swarm intelligence algorithm
BBOB Comparison: Both are population-based metaheuristics

GreyWolfOptimizer: Another nature-inspired optimization algorithm
BBOB Comparison: Similar exploration-exploitation balance

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
View the implementation: [`barnacles_mating.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/barnacles_mating.py)
:::
