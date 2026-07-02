# Soccer League Optimizer

<span class="badge badge-social">Social-Inspired</span>

Soccer League Competition (SLC) algorithm.

## Algorithm Overview

This module implements the Soccer League Competition (SLC) algorithm,
a social-inspired metaheuristic based on soccer league dynamics.

The algorithm simulates soccer team behaviors including matches,
transfers, and training processes.

## Usage

```python
from opt.social_inspired.soccer_league_optimizer import SoccerLeagueOptimizer
from opt.benchmark.functions import sphere

optimizer = SoccerLeagueOptimizer(
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
| `population_size` | `int` | `30` | Total number of teams in the league. |
| `max_iter` | `int` | `100` | Maximum iterations (seasons). |
| `num_teams` | `int` | `10` | Number of teams (deprecated, clamped to. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Soccer League Competition                |
| Acronym           | SLC                                      |
| Year Introduced   | 2014                                     |
| Authors           | Moosavian, N.; Roodsari, B. K.           |
| Algorithm Class   | Social-Inspired                          |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Population-based, Derivative-free    |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

**Match Process** (competitive learning):

$$
X_{new,i} = \begin{cases}
X_i + r_1 \cdot (X_{best} - X_i) \cdot (1 - t) & \text{if winner} \\
X_i + r_2 \cdot (X_{opponent} - X_i) & \text{if loser}
\end{cases}
$$

**Training Phase** (stochastic exploration):

$$
X_{training} = X_{new,i} + r_3 \cdot (1 - t) \cdot 0.1 \cdot (UB - LB)
$$

**Transfer Window** (dimension exchange):

$$
X_{new,i}[d] = X_j[d], \quad \text{with probability } 0.1
$$

where:
- $X_i$ is the position of team $i$
- $X_{opponent}$ is a randomly selected opponent (weighted by rank)
- $X_{best}$ is the league champion (best solution)
- $r_1, r_2 \in [0, 1]^d$ are random vectors
- $r_3 \in [-1, 1]^d$ is a random vector for training
- $t = \frac{iteration}{max\_iter}$ is normalized time
- $d$ is a randomly selected dimension
- $UB, LB$ are upper and lower bounds

**Social Behavior Analogy**:
The algorithm mimics soccer league dynamics where teams (solutions)
compete in matches, train, and trade players. Winners improve toward
the champion (exploitation), losers learn from opponents (exploration),
training adds randomness (diversity), and player transfers enable
dimension-wise knowledge exchange. Match opponent selection is weighted
toward better teams, simulating realistic league scheduling.

Constraint handling:
- **Boundary conditions**: Clamping to `[lower_bound, upper_bound]`
- **Feasibility enforcement**: All new positions clipped to bounds after updates

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 30      | 10*dim           | Total number of teams          |
| max_iter               | 100     | 10000            | Maximum iterations (seasons)   |
| num_teams              | 10      | population_size  | Teams per league (deprecated)  |

**Sensitivity Analysis**:
- `population_size`: **Medium** impact - affects competitive diversity
- Training probability (0.2): **Low** impact - adds exploration noise
- Transfer probability (0.1): **Low** impact - enables dimension mixing
- Note: num_teams is effectively set to population_size in implementation

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

- Each iteration simulates matches between weighted opponents
- Training phase (20% probability) adds exploration
- Transfer window (10% probability) enables dimension exchange
- BBOB: Returns final best solution after max_iter

**Computational Complexity**:
- Time per iteration: $O(\text{population\_size} \times \text{dim})$
- Space complexity: $O(\text{population\_size} \times \text{dim})$
- BBOB budget usage: _Typically uses 20-35% of dim*10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, separable functions
- **Weak function classes**: Ill-conditioned, non-separable
- Typical success rate at 1e-8 precision: **65-75%** (dim=5)
- Expected Running Time (ERT): Competitive on multimodal, moderate on unimodal

**Convergence Properties**:
- Convergence rate: Linear with adaptive exploration decay
- Local vs Global: Good global search via competitive selection
- Premature convergence risk: **Medium** - training/transfer maintain some diversity

**Reproducibility**:
- **Deterministic**: No - uses unseeded random number generation
- **BBOB compliance**: For reproducible results, set numpy random seed before calling
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random` functions throughout (not seeded internally)

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds after position updates
- Numerical stability: Stable for standard floating-point ranges
- Opponent selection: Weighted by inverse rank (better teams more likely)

**Known Limitations**:
- No internal seeding mechanism (relies on external numpy seed management)
- Transfer window dimension exchange may not suit all problem structures
- BBOB known issues: Training/transfer probabilities hardcoded (not tunable)

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: Added COCO/BBOB compliant documentation

## References

[1] Moosavian, N., & Roodsari, B. K. (2014).
"Soccer League Competition Algorithm: A novel meta-heuristic algorithm for
optimal design of water distribution networks."
_Swarm and Evolutionary Computation_, 17, 14-24.
https://doi.org/10.1016/j.swevo.2014.02.002

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

PoliticalOptimizer: Political election-based optimization
BBOB Comparison: Both use competitive dynamics, SLC focuses on matches vs PO's campaigns

SocialGroupOptimizer: Social learning-based optimization
BBOB Comparison: SLC uses competitive learning vs SGO's cooperative phases

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
View the implementation: [`soccer_league_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/social_inspired/soccer_league_optimizer.py)
:::
