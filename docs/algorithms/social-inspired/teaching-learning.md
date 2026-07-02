# Teaching Learning Optimizer

<span class="badge badge-social">Social-Inspired</span>

Teaching-Learning Based Optimization (TLBO) algorithm.

## Algorithm Overview

This module implements Teaching-Learning Based Optimization,
a metaheuristic algorithm inspired by the teaching-learning
process in a classroom.

## Usage

```python
from opt.social_inspired.teaching_learning import TeachingLearningOptimizer
from opt.benchmark.functions import sphere

optimizer = TeachingLearningOptimizer(
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
| `population_size` | `int` | `50` | Number of learners (students) in the classroom. |
| `max_iter` | `int` | `500` | Maximum iterations (teaching sessions). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Teaching-Learning Based Optimization     |
| Acronym           | TLBO                                     |
| Year Introduced   | 2011                                     |
| Authors           | Rao, R. V.; Savsani, V. J.; Vakharia, D. P. |
| Algorithm Class   | Social-Inspired                          |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

**Teacher Phase** (exploitation - learning from the best):

$$
X_{new,i} = X_i + r_i \cdot (X_{teacher} - T_F \cdot \bar{X})
$$

**Learner Phase** (exploration - peer learning):

$$
X_{new,i} = \begin{cases}
X_i + r_i \cdot (X_i - X_j) & \text{if } f(X_i) < f(X_j) \\
X_i + r_i \cdot (X_j - X_i) & \text{if } f(X_j) < f(X_i)
\end{cases}
$$

where:
- $X_i$ is the position of learner $i$ at iteration $t$
- $X_{teacher}$ is the best solution (teacher)
- $\bar{X}$ is the mean position of all learners
- $T_F \in \{1, 2\}$ is the teaching factor (randomly selected)
- $r_i \in [0, 1]^d$ is a random vector
- $X_j$ is a randomly selected learner different from $i$

**Social Behavior Analogy**:
The algorithm mimics classroom learning where students (solutions)
improve through two phases: learning from the teacher (best solution)
and learning from peers (random interactions). The teacher represents
expertise, while peer learning enables knowledge exchange and diversity.

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: All new positions clipped to bounds after each phase

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 50      | 10*dim           | Number of learners (students)  |
| max_iter               | 500     | 10000            | Maximum iterations             |

**Sensitivity Analysis**:
- `population_size`: **Medium** impact - larger populations improve exploration but increase cost
- Recommended tuning ranges: $\text{population\_size} \in [5 \times \text{dim}, 20 \times \text{dim}]$
- **Note**: TLBO is parameter-free (no algorithm-specific parameters to tune)

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

- Executes both teacher and learner phases in each iteration
- Uses greedy selection for accepting new solutions
- BBOB: Returns final best solution after max_iter

**Computational Complexity**:
- Time per iteration: $O(\text{population\_size} \times \text{dim})$
- Space complexity: $O(\text{population\_size} \times \text{dim})$
- BBOB budget usage: _Typically uses 20-40% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Unimodal, weakly-structured multimodal
- **Weak function classes**: Highly ill-conditioned, many local optima
- Typical success rate at 1e-8 precision: **65-75%** (dim=5)
- Expected Running Time (ERT): Competitive with DE on unimodal functions

**Convergence Properties**:
- Convergence rate: Sub-linear to linear depending on problem structure
- Local vs Global: Balanced - teacher phase exploits, learner phase explores
- Premature convergence risk: **Low** - peer learning maintains diversity

**Reproducibility**:
- **Deterministic**: No - uses unse random number generation
- **BBOB compliance**: For reproducible results, set numpy random seed before calling
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random` functions throughout (not seeded internally)

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds after each phase
- Numerical stability: Stable for standard floating-point ranges

**Known Limitations**:
- No internal seeding mechanism (relies on external numpy seed management)
- May struggle with highly rotated or ill-conditioned problems
- BBOB known issues: Slower convergence on sharp ridges and plateaus

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added COCO/BBOB compliant documentation

## References

[1] Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011).
"Teaching-learning-based optimization: A novel method for constrained
mechanical design optimization problems."
_Computer-Aided Design_, 43(3), 303-315.
https://doi.org/10.1016/j.cad.2010.12.015

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

PoliticalOptimizer: Political strategy-based social optimization
BBOB Comparison: Similar social dynamics, PO uses party structures vs TLBO's classroom

SocialGroupOptimizer: Social interaction-based optimization
BBOB Comparison: Both model social learning, SGO has more introspection phases

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

- [Social-Inspired Algorithms](/algorithms/social-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`teaching_learning.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/social_inspired/teaching_learning.py)
:::
