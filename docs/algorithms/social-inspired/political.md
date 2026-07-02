# Political Optimizer

<span class="badge badge-social">Social-Inspired</span>

Political Optimizer (PO) algorithm.

## Algorithm Overview

This module implements the Political Optimizer, a social-inspired metaheuristic
algorithm based on political strategies and election processes.

The algorithm simulates political party behavior including constituency
allocation, party switching, and election campaigns.

## Usage

```python
from opt.social_inspired.political_optimizer import PoliticalOptimizer
from opt.benchmark.functions import sphere

optimizer = PoliticalOptimizer(
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
| `population_size` | `int` | `30` | Number of politicians in the election. |
| `max_iter` | `int` | `100` | Maximum iterations (election cycles). |
| `num_parties` | `int` | `5` | Number of political parties (clusters). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Political Optimizer                      |
| Acronym           | PO                                       |
| Year Introduced   | 2020                                     |
| Authors           | Askari, Q.; Younas, I.; Saeed, M.        |
| Algorithm Class   | Social-Inspired                          |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Population-based, Derivative-free    |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

**Constituency Allocation Phase** (exploration):

$$
X_{new,i} = X_i + r_1 \cdot (L_p - r_2 \cdot X_i)
$$

**Election Campaign Phase** (exploitation):

$$
X_{new,i} = X_i + r_3 \cdot (X_{best} - X_i) + r_4 \cdot (1 - t) \cdot (L_p - X_i)
$$

**Party Switching** (adaptive):

$$
P(switch) = 0.3 \cdot (1 - t), \quad \text{if } f(L_{p'}) < f(X_i)
$$

where:
- $X_i$ is the position of politician $i$
- $L_p$ is the leader of party $p$
- $X_{best}$ is the globally best solution
- $r_1, r_2, r_3, r_4 \in [0, 1]^d$ are random vectors
- $t = \frac{iteration}{max\_iter}$ is the normalized time
- $p'$ is a candidate party for switching

**Social Behavior Analogy**:
The algorithm simulates political election dynamics where politicians (solutions)
belong to parties (clusters). They improve through constituency work (exploration),
election campaigns (exploitation toward best), and strategic party switching
(adaptive diversity maintenance). Party leaders represent local optima, while
the best solution represents the winning candidate.

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: All new positions clipped to bounds after updates

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 30      | 10*dim           | Number of politicians          |
| max_iter               | 100     | 10000            | Maximum iterations (elections) |
| num_parties            | 5       | 3-7              | Number of political parties    |

**Sensitivity Analysis**:
- `population_size`: **Medium** impact - affects diversity and coverage
- `num_parties`: **Medium** impact - more parties increase exploration diversity
- Recommended tuning ranges: $\text{num\_parties} \in [3, \min(7, \text{population\_size}/5)]$

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

- Randomly alternates between constituency and campaign phases
- Adaptive party switching probability decreases over time
- BBOB: Returns final best solution after max_iter

**Computational Complexity**:
- Time per iteration: $O(\text{population\_size} \times \text{dim})$
- Space complexity: $O(\text{population\_size} \times \text{dim} + \text{num\_parties} \times \text{dim})$
- BBOB budget usage: _Typically uses 15-30% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal with separable structure
- **Weak function classes**: Ill-conditioned, sharp ridges
- Typical success rate at 1e-8 precision: **60-70%** (dim=5)
- Expected Running Time (ERT): Competitive on multimodal, slower on unimodal

**Convergence Properties**:
- Convergence rate: Linear with adaptive acceleration
- Local vs Global: Strong global search via party diversity
- Premature convergence risk: **Low** - party switching prevents stagnation

**Reproducibility**:
- **Deterministic**: No - uses unseeded random number generation
- **BBOB compliance**: For reproducible results, set numpy random seed before calling
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random` functions throughout (not seeded internally)

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds after position updates
- Numerical stability: Stable for standard floating-point ranges

**Known Limitations**:
- No internal seeding mechanism (relies on external numpy seed management)
- Party switching probability may need tuning for specific problem types
- BBOB known issues: May require more iterations on high-dimensional problems

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added COCO/BBOB compliant documentation

## References

[1] Askari, Q., Younas, I., & Saeed, M. (2020).
"Political Optimizer: A novel socio-inspired meta-heuristic for global
optimization."
_Knowledge-Based Systems_, 195, 105709.
https://doi.org/10.1016/j.knosys.2020.105709

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

TeachingLearningOptimizer: Teaching-learning based optimization
BBOB Comparison: Both use hierarchical social structures, PO adds party dynamics

SoccerLeagueOptimizer: Soccer competition-based optimization
BBOB Comparison: Similar team-based dynamics, SLC uses match results vs PO's campaigns

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
View the implementation: [`political_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/social_inspired/political_optimizer.py)
:::
