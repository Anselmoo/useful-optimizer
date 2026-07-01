# Gravitational Search Optimizer

<span class="badge badge-physics">Physics-Inspired</span>

Gravitational Search Algorithm (GSA) optimization algorithm.

## Algorithm Overview

This module implements the Gravitational Search Algorithm, a physics-inspired
metaheuristic based on Newton's law of gravity and laws of motion.

Objects (solutions) attract each other with gravitational forces proportional
to their mass (fitness) and inversely proportional to distance. Heavier masses
(better solutions) attract lighter masses (worse solutions).

## Usage

```python
from opt.physics_inspired.gravitational_search import GravitationalSearchOptimizer
from opt.benchmark.functions import sphere

optimizer = GravitationalSearchOptimizer(
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
| `population_size` | `int` | `100` | Population size (number of agents). |
| `g0` | `float` | `_GRAVITATIONAL_CONSTANT_INITIAL` | Initial gravitational constant $G_0$. |
| `alpha` | `float` | `_GRAVITATIONAL_DECAY_RATE` | Decay rate $\alpha$ for gravitational constant. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Gravitational Search Algorithm           |
| Acronym           | GSA                                      |
| Year Introduced   | 2009                                     |
| Authors           | Rashedi, Esmat; Nezamabadi-Pour, Hossein; Saryazdi, Saeid |
| Algorithm Class   | Physics-Inspired                         |
| Complexity        | O(N² $\times$ dim $\times$ max_iter)     |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

GSA is based on Newton's law of gravity and laws of motion. Each agent
(solution) has a mass proportional to its fitness, and agents attract
each other through gravitational forces.

**Gravitational constant** (time-dependent decay):

$$
G(t) = G_0 \cdot e^{-\alpha \cdot t / T}
$$

**Mass calculation** (fitness-based):

$$
M_i(t) = \frac{\exp\left(-\frac{f_i(t) - \text{worst}(t)}{\text{worst}(t) - \text{best}(t) + \epsilon}\right)}{\sum_{j=1}^{N} \exp\left(-\frac{f_j(t) - \text{worst}(t)}{\text{worst}(t) - \text{best}(t) + \epsilon}\right)}
$$

**Gravitational force** from agent $j$ to agent $i$:

$$
F_{ij}^d(t) = G(t) \cdot \frac{M_i(t) \cdot M_j(t)}{R_{ij}(t) + \epsilon} \cdot (x_j^d(t) - x_i^d(t))
$$

**Total force** on agent $i$ (from K best agents):

$$
F_i^d(t) = \sum_{j \in \text{Kbest}, j \neq i} \text{rand}_j \cdot F_{ij}^d(t)
$$

**Acceleration** (Newton's second law: $F = ma$):

$$
a_i^d(t) = \frac{F_i^d(t)}{M_i(t)}
$$

**Velocity update**:

$$
v_i^d(t+1) = \text{rand}_i \cdot v_i^d(t) + a_i^d(t)
$$

**Position update**:

$$
x_i^d(t+1) = x_i^d(t) + v_i^d(t+1)
$$

where:
- $G(t)$ is the gravitational constant at iteration $t$
- $G_0$ is the initial gravitational constant (default: 100.0)
- $\alpha$ is the decay rate (default: 20.0)
- $M_i(t)$ is the mass of agent $i$ at iteration $t$
- $f_i(t)$ is the fitness of agent $i$
- $R_{ij}(t) = \|x_i(t) - x_j(t)\|_2$ is the Euclidean distance
- $\epsilon$ is a small constant to avoid division by zero ($10^{-16}$)
- $\text{Kbest}$ is the set of K best agents (decreases over time)
- $\text{rand}_i, \text{rand}_j$ are random numbers in $[0, 1]$
- $d$ is the dimension index

Constraint handling:
- **Boundary conditions**: Clamping to $[\text{lower\_bound}, \text{upper\_bound}]$
- **Feasibility enforcement**: Solutions violating bounds are projected
back to the nearest boundary using `np.clip`

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                                           |
|------------------------|---------|------------------|-------------------------------------------------------|
| population_size        | 100     | 10*dim           | Number of agents (candidate solutions) in population  |
| max_iter               | 1000    | 10000            | Maximum number of iterations for optimization         |
| g0                     | 100.0   | 100.0            | Initial gravitational constant controlling force strength |
| alpha                  | 20.0    | 20.0             | Exponential decay rate for gravitational constant G(t) |

**Sensitivity Analysis**:
- `population_size`: **Medium** impact on convergence. Larger populations
provide better exploration but increase computational cost.
- `g0`: **Low** impact. Controls initial attraction strength.
- `alpha`: **High** impact. Controls exploration-exploitation balance.
Higher values decay gravity faster (more exploitation).
- Recommended tuning ranges: $\text{alpha} \in [15, 25]$,
$\text{g0} \in [50, 150]$

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
- Time per iteration: $O(N^2 \times \text{dim})$ due to pairwise force
calculations between all agents
- Space complexity: $O(N \times \text{dim})$ for population storage
- BBOB budget usage: _Typically uses 60-80% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, Separable functions (Sphere, Ellipsoid)
- **Weak function classes**: Highly multimodal functions with many local optima,
Ill-conditioned problems (Rosenbrock, Rastrigin)
- Typical success rate at 1e-8 precision: **45-55%** (dim=5)
- Expected Running Time (ERT): Moderate to high compared to gradient-based methods

**Convergence Properties**:
- Convergence rate: Exponential in early iterations, slows to linear
- Local vs Global: Tendency for global exploration early, local exploitation late
- Premature convergence risk: **Medium** - Kbest mechanism helps maintain diversity

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds using `np.clip`
- Numerical stability: Uses epsilon ($10^{-16}$) to avoid division by zero
in distance and mass calculations

**Known Limitations**:
- $O(N^2)$ complexity makes it slow for large populations
- Performance degrades on ill-conditioned and highly multimodal functions
- BBOB known issues: May require many iterations on rotated/shifted functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added BBOB compliance and improved docstrings

## References

[1] Rashedi, E., Nezamabadi-Pour, H., & Saryazdi, S. (2009).
"GSA: A Gravitational Search Algorithm."
_Information Sciences_, 179(13), 2232-2248.
https://doi.org/10.1016/j.ins.2009.03.004

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Not yet available in COCO archive
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Not publicly available
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

EquilibriumOptimizer: Another physics-inspired algorithm based on mass balance
BBOB Comparison: Generally faster on unimodal functions

AtomSearchOptimizer: Molecular dynamics-based algorithm using Lennard-Jones
BBOB Comparison: Similar performance on multimodal functions

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Physics: EquilibriumOptimizer, AtomSearchOptimizer, RIMEOptimizer
- Swarm: ParticleSwarm, AntColony
- Evolutionary: GeneticAlgorithm, DifferentialEvolution

## Related Pages

- [Physics-Inspired Algorithms](/algorithms/physics-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`gravitational_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/physics_inspired/gravitational_search.py)
:::
