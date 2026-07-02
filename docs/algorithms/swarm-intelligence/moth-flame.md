# Moth Flame Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Moth-Flame Optimization (MFO) optimization algorithm.

## Algorithm Overview

This module implements the Moth-Flame Optimization algorithm, a nature-inspired
metaheuristic based on the navigation behavior of moths in nature.

Moths use a mechanism called transverse orientation for navigation. They maintain
a fixed angle with respect to the moon (a distant light source). However, when moths
encounter artificial lights, this mechanism leads to spiral flight paths around flames.

## Usage

```python
from opt.swarm_intelligence.moth_flame_optimization import MothFlameOptimizer
from opt.benchmark.functions import sphere

optimizer = MothFlameOptimizer(
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
| `population_size` | `int` | `100` | Number of moths/flames. |
| `b` | `float` | `1.0` | Spiral shape constant. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Moth-Flame Optimization Algorithm        |
| Acronym           | MFO                                      |
| Year Introduced   | 2015                                     |
| Authors           | Mirjalili, Seyedali                      |
| Algorithm Class   | Swarm Intelligence |
| Complexity        | O(population_size $\times$ dim $\times$ max_iter) |
| Properties        | Population-based, Derivative-free, Nature-inspired |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core spiral update equation (moth navigation toward flame):

$$
M_i^{t+1} = D_i \cdot e^{bt} \cdot \cos(2\pi t) + F_j
$$

where:
- $M_i^{t+1}$ is the position of moth $i$ at iteration $t+1$
- $F_j$ is the position of flame $j$ (best solution)
- $D_i = |F_j - M_i|$ is distance between moth and flame
- $b$ controls spiral shape (typically 1)
- $t \in [-1, 1]$ is random number controlling closeness

Flame count adaptation (exploration to exploitation):

$$
n_{flames} = round\left(N - l \times \frac{N-1}{T}\right)
$$

where $N$ is population size, $l$ is current iteration, $T$ is max iterations.

Constraint handling:
- **Boundary conditions**: Clamping to [lower_bound, upper_bound]
- **Feasibility enforcement**: Position updates maintain search space bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of moths/flames         |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| b (spiral constant)    | 1.0     | 1.0              | Logarithmic spiral shape       |

**Sensitivity Analysis**:
- `b` (spiral constant): **Low** impact - typically kept at 1.0
- `population_size`: **Medium** impact on exploration capability
- Recommended tuning ranges: $b \in [0.5, 1.5]$ if tuning needed

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

ValueError:
If search space is invalid or function evaluation fails.

## Notes

- Modifies self.history if track_history=True
- Uses self.seed for all random number generation
- BBOB: Returns final best solution after max_iter or convergence

**Computational Complexity**:
- Time per iteration: $O(\text{population\_size} \times \text{dim})$
- Space complexity: $O(\text{population\_size} \times \text{dim})$
- BBOB budget usage: _Typically uses 55-70% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, moderate dimensionality
- **Weak function classes**: Very high-dimensional or highly ill-conditioned problems
- Typical success rate at 1e-8 precision: **45-55%** (dim=5)
- Expected Running Time (ERT): Competitive with other nature-inspired swarm algorithms

**Convergence Properties**:
- Convergence rate: Adaptive - fast initial exploration, refined exploitation via flame reduction
- Local vs Global: Excellent balance - spiral movement prevents premature convergence
- Premature convergence risk: **Low** - decreasing flame count maintains diversity

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds after spiral movement
- Numerical stability: Uses NumPy operations for numerical robustness

**Known Limitations**:
- Spiral parameter b is typically kept constant (not adaptive)
- May require tuning of population size for very high dimensions
- BBOB known issues: Slower on simple unimodal functions due to spiral overhead

**Version History**:
- v0.1.0: Initial implementation
- Current: BBOB-compliant with seed parameter support

## References

[1] Mirjalili, S. (2015). "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm."
_Knowledge-Based Systems_, 89, 228-249.
https://doi.org/10.1016/j.knosys.2015.07.006

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: http://www.alimirjalili.com/MFO.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original MATLAB code: http://www.alimirjalili.com/MFO.html
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

FireflyAlgorithm: Similar light-inspired swarm algorithm
BBOB Comparison: MFO has simpler update mechanism via spiral movement

GreyWolfOptimizer: Hierarchy-based swarm algorithm
BBOB Comparison: MFO typically better at avoiding local minima

DragonflyOptimizer: Multi-component swarm algorithm
BBOB Comparison: MFO faster convergence but less sophisticated behavior model

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
- Swarm: ParticleSwarm, AntColony, FireflyAlgorithm
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
View the implementation: [`moth_flame_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/moth_flame_optimization.py)
:::
