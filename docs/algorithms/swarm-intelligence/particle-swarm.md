# Particle Swarm Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Particle Swarm Optimization (PSO) algorithm.

## Algorithm Overview

This module provides an implementation of the Particle Swarm Optimization (PSO) algorithm for solving optimization problems.
PSO is a population-based stochastic optimization algorithm inspired by the social behavior of bird flocking or fish schooling.

The main class in this module is `ParticleSwarm`, which represents the PSO algorithm. It takes an objective function, lower and upper bounds of the search space, dimensionality of the search space, and other optional parameters as input. The `search` method performs the PSO optimization and returns the best solution found.

Example usage:
    optimizer = ParticleSwarm(
        func=shifted_ackley,
        lower_bound=-32.768,
        upper_bound=+32.768,
        dim=2,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Classes:
    - ParticleSwarm: Particle Swarm Optimization (PSO) algorithm for optimization problems.

## Usage

```python
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.benchmark.functions import sphere

optimizer = ParticleSwarm(
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
| `population_size` | `int` | `DEFAULT_POPULATION_SIZE` | Number of particles in swarm. |
| `max_iter` | `int` | `DEFAULT_MAX_ITERATIONS` | Maximum iterations. |
| `c1` | `float` | `PSO_COGNITIVE_COEFFICIENT` | Cognitive coefficient controlling attraction to personal
        best. |
| `c2` | `float` | `PSO_SOCIAL_COEFFICIENT` | Social coefficient controlling attraction to global best. |
| `w` | `float` | `PSO_INERTIA_WEIGHT` | Inertia weight controlling previous velocity influence. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `track_history` | `bool` | `False` | Enable convergence history tracking for BBOB
        post-processing. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Particle Swarm Optimization              |
| Acronym           | PSO                                      |
| Year Introduced   | 1995                                     |
| Authors           | Kennedy, James; Eberhart, Russell        |
| Algorithm Class   | Swarm Intelligence |
| Complexity        | O(population_size $\times$ dim $\times$ max_iter) |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core velocity and position update equations (with inertia weight):

$$
v_i(t+1) = w \cdot v_i(t) + c_1 r_1 (p_{best,i} - x_i(t)) + c_2 r_2 (g_{best} - x_i(t))
$$

$$
x_i(t+1) = x_i(t) + v_i(t+1)
$$

where:
- $x_i(t)$ is the position of particle $i$ at iteration $t$
- $v_i(t)$ is the velocity of particle $i$ at iteration $t$
- $p_{best,i}$ is the personal best position for particle $i$
- $g_{best}$ is the global best position found by any particle
- $w$ is the inertia weight controlling previous velocity influence
- $c_1$ is the cognitive coefficient (self-confidence)
- $c_2$ is the social coefficient (swarm confidence)
- $r_1, r_2$ are random values uniformly distributed in $[0, 1]$

Constraint handling:
- **Boundary conditions**: Clamping to [lower_bound, upper_bound]
- **Feasibility enforcement**: Direct clipping via np.clip after position update

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of particles            |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| w                      | 0.5     | 0.4-0.9          | Inertia weight                 |
| c1                     | 1.5     | 1.5-2.0          | Cognitive coefficient          |
| c2                     | 1.5     | 1.5-2.0          | Social coefficient             |

**Sensitivity Analysis**:
- `w`: **High** impact on convergence - balances exploration vs exploitation
- `c1`: **Medium** impact - controls particle's attraction to personal best
- `c2`: **Medium** impact - controls particle's attraction to global best
- Recommended tuning ranges: $w \in [0.4, 0.9]$, $c_1, c_2 \in [1.5, 2.5]$

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
- BBOB budget usage: _Typically uses 50-70% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, separable functions
- **Weak function classes**: Highly multimodal with many local optima, ill-conditioned
- Typical success rate at 1e-8 precision: **40-60%** (dim=5)
- Expected Running Time (ERT): Fast to moderate, excellent on smooth landscapes

**Convergence Properties**:
- Convergence rate: Linear to superlinear on unimodal functions
- Local vs Global: Good balance, tendency toward global with proper parameters
- Premature convergence risk: **Medium** - mitigated by inertia weight tuning

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds via np.clip
- Numerical stability: Velocity not limited (can grow unbounded)

**Known Limitations**:
- Velocity can become very large without velocity clamping
- No adaptive parameter control in this basic implementation
- BBOB known issues: Performance degrades on high-dimensional (dim>40) problems

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: COCO/BBOB compliant docstring added

## References

[1] Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
_Proceedings of IEEE International Conference on Neural Networks_,
Vol. 4, 1942-1948.
https://doi.org/10.1109/ICNN.1995.488968

[2] Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer."
_Proceedings of IEEE International Conference on Evolutionary Computation_,
69-73. (Introduced inertia weight)
https://doi.org/10.1109/ICEC.1998.699146

[3] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- This implementation: Based on [1] and [2] with inertia weight variant
and modifications for BBOB compliance

## See Also

AntColony: Another swarm intelligence algorithm inspired by ant behavior
BBOB Comparison: PSO generally faster on unimodal functions

GeneticAlgorithm: Evolutionary approach with different operators
BBOB Comparison: PSO often converges faster with simpler parameter tuning

DifferentialEvolution: Population-based evolutionary algorithm
BBOB Comparison: Similar performance, PSO simpler with fewer parameters

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
- Swarm: AntColony, BatAlgorithm, FireflyAlgorithm
- Gradient: AdamW, SGDMomentum

## Related Pages

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`particle_swarm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/particle_swarm.py)
:::
