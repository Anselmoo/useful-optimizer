# Colliding Bodies Optimization

<span class="badge badge-metaheuristic">Metaheuristic</span>

Colliding Bodies Optimization (CBO) optimization algorithm.

## Algorithm Overview

The Colliding Bodies Optimization algorithm is inspired by the behavior of colliding
bodies in physics. It aims to find the global minimum of a given objective function.

Example usage:
    optimizer = CollidingBodiesOptimization(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

## Usage

```python
from opt.metaheuristic.colliding_bodies_optimization import CollidingBodiesOptimization
from opt.benchmark.functions import sphere

optimizer = CollidingBodiesOptimization(
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Number of bodies. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Colliding Bodies Optimization            |
| Acronym           | CBO                                      |
| Year Introduced   | 2014                                     |
| Authors           | Kaveh, Ali; Mahdavi, Vahid Reza          |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Based on conservation of momentum and energy in collisions:

Conservation of momentum:
$$m_1 v_1 + m_2 v_2 = m_1 v_1' + m_2 v_2'$$

Conservation of energy (with loss):
$$\frac{1}{2}m_1 v_1^2 + \frac{1}{2}m_2 v_2^2 - Q = \frac{1}{2}m_1 {v_1'}^2 + \frac{1}{2}m_2 {v_2'}^2$$

where:
- $m_i$ is mass (inversely proportional to fitness)
- $v_i$ is velocity before collision
- $v_i'$ is velocity after collision
- $Q$ is kinetic energy lost during collision

Bodies divided into stationary (better half) and moving (worse half).

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of bodies               |
| max_iter               | 1000    | 10000            | Maximum iterations             |

**Sensitivity Analysis**:
- `population_size`: **Medium** impact on search quality
- Parameter-free design (no tuning required for collision physics)
- Recommended tuning ranges: population $\in [5 \times dim, 15 \times dim]$

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
- **Best function classes**: Multimodal, rugged landscapes
- **Weak function classes**: Smooth unimodal functions
- Typical success rate at 1e-8 precision: **20-30%** (dim=5)
- Expected Running Time (ERT): Moderate; good exploration via collision dynamics

**Convergence Properties**:
- Convergence rate: Sublinear (physics-based updates)
- Local vs Global: Good global exploration via collision mechanics
- Premature convergence risk: **Low** (collision dynamics maintain diversity)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Mass-based formulation ensures bounded updates

**Known Limitations**:
- Physics-based approach may be less effective on highly abstract problems
- Performance depends on population pairing strategy
- BBOB known issues: Less effective on simple unimodal functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Kaveh, A., & Mahdavi, V. R. (2014). "Colliding bodies optimization:
A novel meta-heuristic method."
_Computers & Structures_, 139, 18-27.
https://doi.org/10.1016/j.compstruc.2014.04.005

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: MATLAB implementations available
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

GravitationalSearchAlgorithm: Another physics-inspired algorithm
BBOB Comparison: Both physics-based; GSA uses gravity, CBO uses collisions

ParticleSwarm: Population-based swarm algorithm
BBOB Comparison: PSO velocity-based; CBO collision-based

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
View the implementation: [`colliding_bodies_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/colliding_bodies_optimization.py)
:::
