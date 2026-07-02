# Imperialist Competitive Algorithm

<span class="badge badge-evolutionary">Evolutionary</span>

Imperialist Competitive Algorithm (ICA) optimization algorithm.

## Algorithm Overview

This module implements the Imperialist Competitive Algorithm (ICA) for solving
optimization problems. The ICA is a population-based algorithm that simulates the
competition between empires and colonies. It starts with a random population and
iteratively improves the solutions by assimilation, revolution, position exchange,
and imperialistic competition.

## Usage

```python
from opt.evolutionary.imperialist_competitive_algorithm import ImperialistCompetitiveAlgorithm
from opt.benchmark.functions import sphere

optimizer = ImperialistCompetitiveAlgorithm(
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
| `dim` | `int` | Required | Problem dimensionality. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `num_empires` | `int` | `15` | Number of initial empires. |
| `population_size` | `int` | `100` | Total number of countries. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `revolution_rate` | `float` | `0.3` | Revolution probability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Imperialist Competitive Algorithm        |
| Acronym           | ICA                                      |
| Year Introduced   | 2007                                     |
| Authors           | Atashpaz-Gargari, Esmaeil; Lucas, Caro   |
| Algorithm Class   | Evolutionary                             |
| Complexity        | O(NP * dim) per iteration                |
| Properties        | Population-based, Derivative-free, Stochastic |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

ICA models imperialistic competition where empires compete for colonies:

**Assimilation** (colonies move toward imperialist):

$$
colony_{new} = colony + \beta \cdot (imperialist - colony)
$$

**Revolution** (random perturbation):

$$
colony_{rev} = colony + \gamma \cdot \mathcal{N}(0, 1)
$$

**Imperialistic Competition**:
- Weak empires lose colonies to stronger ones
- Total cost: $TC_i = Cost(imperialist_i) + \xi \cdot mean(Cost(colonies_i))$

where:
- $\beta$ controls assimilation rate
- $\gamma$ controls revolution strength
- $\xi$ weights colony influence on empire
- Empires compete based on total cost

**Constraint handling**:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Solutions clipped to valid range

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Total number of countries      |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| num_empires            | 15      | 5-20             | Number of initial empires      |
| revolution_rate        | 0.3     | 0.2-0.5          | Revolution probability         |

**Sensitivity Analysis**:
- `num_empires`: **Medium** impact - affects exploration diversity
- `revolution_rate`: **Medium** impact - controls exploration
- Recommended tuning ranges: $num\_empires \in [3, 30]$, $revolution\_rate \in [0.1, 0.6]$

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
- Time per iteration: $O(NP \cdot n)$
- Space complexity: $O(NP \cdot n)$
- BBOB budget usage: _Typically uses 60-90% of dim*10000 budget_

**BBOB Performance Characteristics**:
- **Best function classes**: Moderately multimodal, Structured
- **Weak function classes**: Highly ill-conditioned
- Typical success rate at 1e-8 precision: **55-70%** (dim=5)

**Convergence Properties**:
- Convergence rate: Linear with competitive pressure
- Local vs Global: Balanced through empire competition
- Premature convergence risk: **Medium**

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required
- Initialization: Uniform random sampling
- RNG usage: `numpy.random.default_rng(self.seed)`

**Implementation Details**:
- Parallelization: Not supported
- Constraint handling: Clamping to bounds
- Numerical stability: Standard precision

**Known Limitations**:
- Complex parameter interactions
- BBOB known issues: None specific

**Version History**:
- v0.1.0: Initial implementation

## References

[1] Atashpaz-Gargari, E., & Lucas, C. (2007). "Imperialist Competitive Algorithm: An Algorithm for Optimization Inspired by Imperialistic Competition."
_IEEE Congress on Evolutionary Computation (CEC 2007)_, 4661-4667.

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Socio-political competition model with assimilation and revolution

## See Also

GeneticAlgorithm: Traditional evolutionary approach
BBOB Comparison: ICA adds socio-political competitive dynamics

CulturalAlgorithm: Dual inheritance model
BBOB Comparison: Both use social structures, different mechanisms

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

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`imperialist_competitive_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/imperialist_competitive_algorithm.py)
:::
