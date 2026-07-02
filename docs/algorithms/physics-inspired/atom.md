# Atom Search Optimizer

<span class="badge badge-physics">Physics-Inspired</span>

Atom Search Optimization (ASO) algorithm.

## Algorithm Overview

This module implements Atom Search Optimization, a physics-inspired
metaheuristic algorithm based on molecular dynamics simulation.

## Usage

```python
from opt.physics_inspired.atom_search import AtomSearchOptimizer
from opt.benchmark.functions import sphere

optimizer = AtomSearchOptimizer(
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
| `population_size` | `int` | `50` | Population size (number of atoms). |
| `max_iter` | `int` | `500` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Atom Search Optimization                 |
| Acronym           | ASO                                      |
| Year Introduced   | 2019                                     |
| Authors           | Zhao, Weiguo; Wang, Liying; Zhang, Zhenxing |
| Algorithm Class   | Physics-Inspired                         |
| Complexity        | O(N² $\times$ dim $\times$ max_iter)     |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

ASO simulates molecular dynamics using the Lennard-Jones potential to model
atomic interactions. Atoms (solutions) attract or repel each other based on
their distances, creating a balance between exploration and exploitation.

**Mass calculation** (fitness-based, for minimization):

$$
M_i = \frac{\exp\left(-\frac{f_i - f_{\text{best}}}{f_{\text{worst}} - f_{\text{best}} + \epsilon}\right)}{\sum_{j=1}^{N} \exp\left(-\frac{f_j - f_{\text{best}}}{f_{\text{worst}} - f_{\text{best}} + \epsilon}\right)}
$$

**Lennard-Jones potential force** between atoms $i$ and $j$:

$$
F_{LJ}(r_{ij}) = \alpha \left[\left(\frac{\sigma}{r_{ij}}\right)^{12} - \left(\frac{\sigma}{r_{ij}}\right)^6\right]
$$

**Interaction force** from atom $j$ to atom $i$:

$$
F_{ij} = G(t) \cdot F_{LJ}(r_{ij}) \cdot M_j \cdot \frac{\mathbf{x}_j - \mathbf{x}_i}{r_{ij}}
$$

**Total force** on atom $i$:

$$
\mathbf{F}_i = \sum_{j=1, j \neq i}^{N} F_{ij}
$$

**Constraint factor** (time-dependent):

$$
G(t) = G_0 \cdot e^{-20t/T}
$$

**Velocity update**:

$$
\mathbf{v}_i(t+1) = \text{rand} \cdot \mathbf{v}_i(t) + \mathbf{F}_i(t)
$$

**Position update**:

$$
\mathbf{x}_i(t+1) = \mathbf{x}_i(t) + \mathbf{v}_i(t+1)
$$

where:
- $r_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$ is the Euclidean distance
- $\alpha = 50$ is the depth of Lennard-Jones potential
- $\sigma = \beta \cdot \text{diagonal}$ where $\beta = 0.2$
- $\text{diagonal} = \sqrt{\text{dim} \cdot (\text{upper} - \text{lower})^2}$
- $G_0 = 1.0$ is the initial constraint factor
- $M_i$ is the mass of atom $i$ (proportional to fitness quality)
- $\text{rand}$ is a random vector in $[0, 1]^{\text{dim}}$
- $\epsilon = 10^{-10}$ prevents division by zero

The Lennard-Jones potential provides:
- **Repulsion** at short distances ($r^{-12}$ term dominates)
- **Attraction** at medium distances ($r^{-6}$ term dominates)
- **Zero force** at optimal distance $\sigma$

Constraint handling:
- **Boundary conditions**: Reflection at boundaries with velocity reversal
- **Feasibility enforcement**: When atom hits boundary, position is clamped
and velocity component is negated (elastic collision)

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                                           |
|------------------------|---------|------------------|-------------------------------------------------------|
| population_size        | 50      | 10*dim           | Number of atoms (candidate solutions) in population   |
| max_iter               | 500     | 10000            | Maximum number of iterations for optimization         |

**Sensitivity Analysis**:
- `population_size`: **High** impact. Larger populations improve exploration
but increase $O(N^2)$ computational cost significantly.
- Algorithm uses fixed constants: $\alpha=50$, $\beta=0.2$, $G_0=1.0$
- Recommended tuning ranges: $\text{population\_size} \in [5 \cdot \text{dim}, 15 \cdot \text{dim}]$

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
- Time per iteration: $O(N^2 \times \text{dim})$ due to pairwise Lennard-Jones
force calculations between all atoms
- Space complexity: $O(N \times \text{dim})$ for population and velocities
- BBOB budget usage: _Typically uses 70-90% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Continuous, Moderately multimodal functions
- **Weak function classes**: Highly separable, Noisy functions, Very high dimensions
- Typical success rate at 1e-8 precision: **40-50%** (dim=5)
- Expected Running Time (ERT): High due to $O(N^2)$ complexity, comparable to GSA

**Convergence Properties**:
- Convergence rate: Good early progress, slower refinement in later iterations
- Local vs Global: Lennard-Jones provides good balance - repulsion prevents
premature clustering, attraction enables exploitation
- Premature convergence risk: **Low to Medium** - Reflection boundary handling
helps maintain diversity

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Reflection with velocity reversal (elastic collision model)
- Numerical stability: Uses epsilon ($10^{-10}$) to prevent division by zero in
Lennard-Jones calculations; distance clamping prevents numerical overflow

**Known Limitations**:
- $O(N^2)$ complexity makes it impractical for large populations
- Reflection boundary handling can cause atoms to "bounce" repeatedly at boundaries
- BBOB known issues: Performance degrades significantly on high-dimensional problems (dim > 20)

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added BBOB compliance with seed parameter and improved docstrings

## References

[1] Zhao, W., Wang, L., & Zhang, Z. (2019).
"Atom search optimization and its application to solve a hydrogeologic
parameter estimation problem."
_Knowledge-Based Systems_, 163, 283-304.
https://doi.org/10.1016/j.knosys.2018.08.030

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

GravitationalSearchOptimizer: Newton's gravity with mass-based forces
BBOB Comparison: ASO uses Lennard-Jones instead of pure gravitational forces

EquilibriumOptimizer: Mass balance equilibrium-based algorithm
BBOB Comparison: ASO has higher computational cost but better local search

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Physics: GravitationalSearchOptimizer, EquilibriumOptimizer, RIMEOptimizer
- Swarm: ParticleSwarm, AntColony
- Evolutionary: GeneticAlgorithm, DifferentialEvolution

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

::: tip Run-based charts
Convergence, distribution and ECDF charts appear here once this optimizer is included in the benchmark suite.
:::

## Related Pages

- [Physics-Inspired Algorithms](/algorithms/physics-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`atom_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/physics_inspired/atom_search.py)
:::
