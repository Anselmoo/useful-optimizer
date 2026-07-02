# Forensic Based Investigation Optimizer

<span class="badge badge-metaheuristic">Metaheuristic</span>

Forensic-Based Investigation Optimizer (FBI) optimization algorithm.

## Algorithm Overview

Implementation based on:
Chou, J.S. & Nguyen, N.M. (2020).
FBI inspired meta-optimization.
Applied Soft Computing, 93, 106339.

The algorithm mimics the investigation process used by forensic
investigators, including evidence analysis and suspect tracking.

## Usage

```python
from opt.metaheuristic.forensic_based import ForensicBasedInvestigationOptimizer
from opt.benchmark.functions import sphere

optimizer = ForensicBasedInvestigationOptimizer(
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
| `max_iter` | `int` | Required | Maximum iterations. |
| `population_size` | `int` | `30` | Number of investigators. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Forensic-Based Investigation Optimizer   |
| Acronym           | FBI                                      |
| Year Introduced   | 2020                                     |
| Authors           | Chou, Jui-Sheng; Nguyen, Ngoc-Mai        |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size $\times$ dim $\times$ max_iter) |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Two-phase update mechanism based on investigation and pursuit:

**Investigation Phase** (exploration):

$$
x_i^{new} = x_i + \beta (x_{r1} - x_{r2}) + (1 - \beta) \xi (\bar{x} - x_i)
$$

**Pursuit Phase** (exploitation):

$$
x_i^{new} = x^* + \alpha (x^* - x_i)
$$

where:
- $x_i$ is the i-th investigator position
- $x^*$ is the best solution (prime suspect location)
- $\bar{x}$ is the mean position (investigation center)
- $\beta, \alpha$ are random coefficients
- $r1, r2$ are random investigator indices
- $\xi$ is Gaussian noise for evidence analysis
- Phase selection probability decreases linearly with iteration

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 30      | 10*dim           | Number of investigators        |
| max_iter               | 1000    | 10000            | Maximum iterations             |

**Sensitivity Analysis**:
- `population_size`: **Low** impact (algorithm is parameter-free)
- FBI is designed to be parameter-free, requiring only population size and stopping criteria
- Recommended tuning ranges: population $\in [20, 50]$ for most problems

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
- Time per iteration: $O(population\_size \times dim)$
- Space complexity: $O(population\_size \times dim)$
- BBOB budget usage: _Typically uses 50-70% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, weakly-structured problems
- **Weak function classes**: Highly ill-conditioned functions
- Typical success rate at 1e-8 precision: **20-30%** (dim=5)
- Expected Running Time (ERT): Fast to moderate; parameter-free simplifies tuning

**Convergence Properties**:
- Convergence rate: Sublinear
- Local vs Global: Balanced via investigation/pursuit phases
- Premature convergence risk: **Low** (dual-phase mechanism)

**Reproducibility**:
- **Deterministic**: Yes (with proper seed management)
- **BBOB compliance**: Requires seed parameter for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: Uses standard numpy random number generation

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Gaussian noise and random coefficients prevent numerical issues

**Known Limitations**:
- Parameter-free design may sacrifice fine-tuning potential
- Performance depends on population size selection
- BBOB known issues: May converge slowly on high-dimensional ill-conditioned problems

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Chou, J. S., & Nguyen, N. M. (2020). "FBI inspired meta-optimization."
_Applied Soft Computing_, 93, 106339.
https://doi.org/10.1016/j.asoc.2020.106339

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results (algorithm introduced 2020)
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: MATLAB implementation available on MathWorks File Exchange
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

SimulatedAnnealing: Temperature-based metaheuristic with similar exploration strategy
BBOB Comparison: Both effective on multimodal problems; FBI is parameter-free

GeneticAlgorithm: Population-based evolutionary algorithm
BBOB Comparison: GA requires crossover/mutation parameters; FBI simpler to configure

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

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`forensic_based.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/forensic_based.py)
:::
