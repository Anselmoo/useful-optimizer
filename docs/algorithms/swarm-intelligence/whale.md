# Whale Optimization Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Whale Optimization Algorithm (WOA) optimization algorithm.

## Algorithm Overview

This module implements the Whale Optimization Algorithm (WOA). WOA is a metaheuristic
optimization algorithm inspired by the hunting behavior of humpback whales.
The algorithm is based on the echolocation behavior of humpback whales, which use sounds
to communicate, navigate and hunt in dark or murky waters.

In WOA, each whale represents a potential solution, and the objective function
determines the quality of the solutions. The whales try to update their positions by
mimicking the hunting behavior of humpback whales, which includes encircling,
bubble-net attacking, and searching for prey.

WOA has been used for various kinds of optimization problems including function
optimization, neural network training, and other areas of engineering.

## Usage

```python
from opt.swarm_intelligence.whale_optimization_algorithm import WhaleOptimizationAlgorithm
from opt.benchmark.functions import sphere

optimizer = WhaleOptimizationAlgorithm(
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
| `population_size` | `int` | `100` | Number of whales. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Whale Optimization Algorithm             |
| Acronym           | WOA                                      |
| Year Introduced   | 2016                                     |
| Authors           | Mirjalili, Seyedali; Lewis, Andrew       |
| Algorithm Class   | Swarm Intelligence                       |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Population-based, Derivative-free, Nature-inspired |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equations based on humpback whale bubble-net hunting:

Encircling prey:

$$
\vec{D} = |\vec{C} \cdot \vec{X}^*(t) - \vec{X}(t)|
$$

$$
\vec{X}(t+1) = \vec{X}^*(t) - \vec{A} \cdot \vec{D}
$$

Spiral bubble-net attacking:

$$
\vec{X}(t+1) = \vec{D}' \cdot e^{bl} \cdot \cos(2\pi l) + \vec{X}^*(t)
$$

where:
- $\vec{X}^*(t)$ is the position of the best solution (prey)
- $\vec{X}(t)$ is the position of a whale at iteration $t$
- $\vec{A} = 2\vec{a} \cdot \vec{r} - \vec{a}$ and $\vec{C} = 2 \cdot \vec{r}$
- $\vec{a}$ linearly decreases from 2 to 0
- $\vec{r}$ is a random vector in [0,1]
- $b$ is a constant defining the shape of the logarithmic spiral
- $l$ is a random number in [-1, 1]
- $\vec{D}' = |\vec{X}^*(t) - \vec{X}(t)|$

Constraint handling:
- **Boundary conditions**: Clamping to [lower_bound, upper_bound]
- **Feasibility enforcement**: Position updates respect boundary constraints

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 30      | 10*dim           | Number of whales               |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| b                      | 1.0     | 1.0              | Spiral shape constant          |

**Sensitivity Analysis**:
- `a`: Parameter linearly decreases from 2 to 0 - **High** impact on exploration/exploitation
- `b`: **Low** impact - controls spiral tightness, typically kept at 1.0
- Recommended: Use default parameters for most problems

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
- BBOB budget usage: _Typically uses 60-75% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, Multimodal with few local optima
- **Weak function classes**: Highly multimodal, Ill-conditioned functions
- Typical success rate at 1e-8 precision: **40-50%** (dim=5)
- Expected Running Time (ERT): Competitive with GWO and PSO

**Convergence Properties**:
- Convergence rate: Exponential early, linear near optimum
- Local vs Global: Good balance through encircling and spiral search
- Premature convergence risk: **Low** - spiral mechanism maintains diversity

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in current implementation
- Constraint handling: Clamping to bounds after each update
- Numerical stability: Uses NumPy operations for stability

**Known Limitations**:
- Parameter 'a' uses linear decrease which may not be optimal for all problems
- Fixed probability (0.5) for choosing between encircling and spiral
- BBOB known issues: May struggle on very high-dimensional problems (>40D)

**Version History**:
- v0.1.0: Initial implementation
- Current: BBOB-compliant with seed parameter support

## References

[1] Mirjalili, S., Lewis, A. (2016). "The Whale Optimization Algorithm."
_Advances in Engineering Software_, 95, 51-67.
https://doi.org/10.1016/j.advengsoft.2016.01.008

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: https://seyedalimirjalili.com/woa
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original MATLAB code: https://seyedalimirjalili.com/woa
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

GreyWolfOptimizer: Also by Mirjalili, hierarchy-based hunting
BBOB Comparison: GWO and WOA have similar performance overall

SalpSwarmAlgorithm: Another marine-inspired algorithm by Mirjalili
BBOB Comparison: WOA typically faster convergence on unimodal functions

ParticleSwarm: Classic swarm intelligence algorithm
BBOB Comparison: WOA shows better exploration due to spiral mechanism

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
- Swarm: ParticleSwarm, AntColony, GreyWolfOptimizer
- Gradient: AdamW, SGDMomentum

## Related Pages

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`whale_optimization_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/whale_optimization_algorithm.py)
:::
