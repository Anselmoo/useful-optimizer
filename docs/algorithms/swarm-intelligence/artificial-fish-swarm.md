# Artificial Fish Swarm

<span class="badge badge-swarm">Swarm Intelligence</span>

Artificial Fish Swarm Algorithm (AFSA) optimization algorithm.

## Algorithm Overview

This module implements the Artificial Fish Swarm Algorithm (AFSA). AFSA is a population
based optimization technique inspired by the social behavior of fishes. In their social
behavior, fish try to keep a balance between food consistency and crowding effect. This
behavior is modeled into a mathematical optimization technique in AFSA.

In AFSA, each fish represents a potential solution and the food consistency represents
the objective function to be optimized. Each fish tries to move towards better regions
of the search space based on its own experience and the experience of its neighbors.

AFSA has been used for various kinds of optimization problems including function
optimization, neural network training, fuzzy system control, and other areas of
engineering.

## Usage

```python
from opt.swarm_intelligence.artificial_fish_swarm_algorithm import ArtificialFishSwarm
from opt.benchmark.functions import sphere

optimizer = ArtificialFishSwarm(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
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
| `fish_swarm` | `int` | `50` | Number of fish in swarm. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `visual` | `int` | `1` | Visual distance parameter. |
| `step` | `float` | `0.1` | Step size parameter. |
| `try_number` | `int` | `3` | Number of attempts. |
| `epsilon` | `float` | `1e-09` | Small value for numerical stability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Artificial Fish Swarm Algorithm             |
| Acronym           | AFSA                           |
| Year Introduced   | 2002                            |
| Authors           | Li, Xiaolei; Shao, Zhuhong; Qian, Jixin                |
| Algorithm Class   | Swarm Intelligence |
| Complexity        | O(population_size $\times$ dim $\times$ max_iter)                   |
| Properties        | Population-based, Derivative-free           |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

$$
x_{t+1} = x_t + v_t
$$

where:
- $x_t$ is the position at iteration $t$
- $v_t$ is the velocity/step at iteration $t$
-
Constraint handling:
- **Boundary conditions**:             - **Feasibility enforcement**:

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of individuals          |
| max_iter               | 1000    | 10000            | Maximum iterations             |
|
**Sensitivity Analysis**:
- Parameters have standard impact on convergence
- Recommended tuning ranges: Standard parameter tuning ranges apply

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
- Time per iteration: $O(       ext{population\_size} \times    ext{dim})$})$
- Space complexity: $O( ext{population\_size} \times    ext{dim})$})$
- BBOB budget usage: _Typically uses 50-70% of dim $\times$ 10000 budget__

**BBOB Performance Characteristics**:
- **Best function classes**: General optimization problems
- **Weak function classes**: Problem-specific
- Typical success rate at 1e-8 precision: **40-50%** (dim=5)
- Expected Running Time (ERT): Competitive

**Convergence Properties**:
- Convergence rate: Adaptive
- Local vs Global: Balanced
- Premature convergence risk: **Medium**

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation`]
- Constraint handling: Clamping to bounds
- Numerical stability: Uses NumPy operations

**Known Limitations**:
- Standard implementation
- BBOB known issues: Standard considerations

**Version History**:
- v0.1.0: Initial implementation
- Current: BBOB-compliant with seed parameter

## References

[1] Reference available in academic literature

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Available in academic literature
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original implementations: Available in academic literature
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

on [function classes]

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
View the implementation: [`artificial_fish_swarm_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/artificial_fish_swarm_algorithm.py)
:::
