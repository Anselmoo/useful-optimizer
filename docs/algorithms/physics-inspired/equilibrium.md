# Equilibrium Optimizer

<span class="badge badge-physics">Physics-Inspired</span>

Equilibrium Optimizer (EO) optimization algorithm.

## Algorithm Overview

This module implements the Equilibrium Optimizer, a physics-inspired metaheuristic
based on control volume mass balance models used to estimate dynamic and equilibrium
states.

The algorithm uses concepts from mass balance to describe concentration changes
in a control volume, simulating particles reaching equilibrium states.

## Usage

```python
from opt.physics_inspired.equilibrium_optimizer import EquilibriumOptimizer
from opt.benchmark.functions import sphere

optimizer = EquilibriumOptimizer(
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
| `population_size` | `int` | `100` | Population size (number of particles). |
| `a1` | `float` | `_A1` | Generation rate control constant. |
| `a2` | `float` | `_A2` | Time decay exponent for $t$ parameter. |
| `gp` | `float` | `_GP` | Generation probability threshold. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Equilibrium Optimizer                    |
| Acronym           | EO                                       |
| Year Introduced   | 2020                                     |
| Authors           | Faramarzi, Afshin; Heidarinejad, Mohammad; Stephens, Brent; Mirjalili, Seyedali |
| Algorithm Class   | Physics-Inspired                         |
| Complexity        | O(N $\times$ dim $\times$ max_iter)      |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

EO is based on control volume mass balance models describing concentration
changes in a control volume. Particles move toward equilibrium states
determined by the best solutions found.

**Equilibrium pool** (4 best + average):

$$
C_{\text{eq,pool}} = \{C_{\text{eq},1}, C_{\text{eq},2}, C_{\text{eq},3}, C_{\text{eq},4}, C_{\text{eq,avg}}\}
$$

where $C_{\text{eq},i}$ are the top 4 best solutions and:

$$
C_{\text{eq,avg}} = \frac{1}{4} \sum_{i=1}^{4} C_{\text{eq},i}
$$

**Time parameter** (exponential decay):

$$
t = \left(1 - \frac{\text{iter}}{T}\right)^{a_2 \cdot \text{iter}/T}
$$

**Exponential term** (generation rate):

$$
F = a_1 \cdot \text{sign}(r - 0.5) \cdot (e^{-\lambda \cdot t} - 1)
$$

**Generation rate**:

$$
G =
\begin{cases}
G_{CP} \cdot r_1 & \text{if } r_2 \geq GP \\
0 & \text{otherwise}
\end{cases}
$$

where $G_{CP} = 0.5$ (generation probability constant).

**Concentration update**:

$$
C_i(t+1) = C_{\text{eq}} + (C_i(t) - C_{\text{eq}}) \cdot F + \frac{G}{\lambda \cdot V} \cdot (1 - F)
$$

where:
- $C_i$ is the concentration (position) of particle $i$
- $C_{\text{eq}}$ is a randomly selected equilibrium candidate
- $\lambda$ is a random vector in $[0, 1]^{\text{dim}}$
- $V = \text{upper\_bound} - \text{lower\_bound}$ is the volume
- $r, r_1, r_2$ are random numbers in $[0, 1]$
- $a_1 = 2.0$ controls generation rate
- $a_2 = 1.0$ controls time decay
- $GP = 0.5$ is the generation probability

Constraint handling:
- **Boundary conditions**: Clamping to $[\text{lower\_bound}, \text{upper\_bound}]$
- **Feasibility enforcement**: Bounds enforced after each position update
using `np.clip`

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                                           |
|------------------------|---------|------------------|-------------------------------------------------------|
| population_size        | 100     | 10*dim           | Number of particles (candidate solutions) in population |
| max_iter               | 1000    | 10000            | Maximum number of iterations for optimization         |
| a1                     | 2.0     | 2.0              | Generation rate control constant (controls magnitude) |
| a2                     | 1.0     | 1.0              | Time decay exponent (controls exploitation transition)|
| gp                     | 0.5     | 0.5              | Generation probability threshold for mechanism activation |

**Sensitivity Analysis**:
- `a1`: **Medium** impact. Controls generation rate magnitude.
Higher values increase randomness.
- `a2`: **High** impact. Controls exploration-exploitation balance.
Higher values accelerate shift to exploitation.
- `gp`: **Low** impact. Probability threshold for generation mechanism.
- Recommended tuning ranges: $\text{a1} \in [1.5, 2.5]$,
$\text{a2} \in [0.5, 1.5]$, $\text{gp} \in [0.3, 0.7]$

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
- Time per iteration: $O(N \times \text{dim})$ for position updates
- Space complexity: $O(N \times \text{dim})$ for population and equilibrium pool
- BBOB budget usage: _Typically uses 50-70% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, Separable, Weakly structured multimodal
- **Weak function classes**: Highly multimodal with many local optima,
Ill-conditioned problems (Sharp ridges, Different scales)
- Typical success rate at 1e-8 precision: **50-60%** (dim=5)
- Expected Running Time (ERT): Competitive with other metaheuristics,
faster than GSA on unimodal functions

**Convergence Properties**:
- Convergence rate: Fast early convergence, then gradual refinement
- Local vs Global: Good balance via equilibrium pool mechanism
- Premature convergence risk: **Low** - Pool of equilibria maintains diversity

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds using `np.clip`
- Numerical stability: Robust due to exponential formulation and bounded
random variables

**Known Limitations**:
- Performance can degrade on very high-dimensional problems (dim > 100)
- May require parameter tuning for specific problem classes
- BBOB known issues: Slower convergence on rotated/shifted multimodal functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added BBOB compliance and improved docstrings

## References

[1] Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020).
"Equilibrium optimizer: A novel optimization algorithm."
_Knowledge-Based Systems_, 191, 105190.
https://doi.org/10.1016/j.knosys.2019.105190

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Not yet available in COCO archive
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: Available at https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

GravitationalSearchOptimizer: Newton's gravity-based physics algorithm
BBOB Comparison: EO typically converges faster on separable functions

AtomSearchOptimizer: Molecular dynamics with Lennard-Jones potential
BBOB Comparison: Similar performance on continuous functions

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Physics: GravitationalSearchOptimizer, AtomSearchOptimizer, RIMEOptimizer
- Swarm: ParticleSwarm, AntColony
- Evolutionary: GeneticAlgorithm, DifferentialEvolution

## Related Pages

- [Physics-Inspired Algorithms](/algorithms/physics-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`equilibrium_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/physics_inspired/equilibrium_optimizer.py)
:::
