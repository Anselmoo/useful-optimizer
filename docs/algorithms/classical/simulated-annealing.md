# Simulated Annealing

<span class="badge badge-classical">Classical</span>

Simulated Annealing (SA) metaheuristic optimization algorithm.

## Algorithm Overview

This module provides an implementation of the Simulated Annealing optimization
algorithm. Simulated Annealing is a metaheuristic optimization algorithm that is
inspired by the annealing process in metallurgy. It is used to find the global minimum
of a given objective function in a search space.

## Usage

```python
from opt.classical.simulated_annealing import SimulatedAnnealing
from opt.benchmark.functions import sphere

optimizer = SimulatedAnnealing(
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
| `population_size` | `int` | `100` | Number of independent runs. |
| `max_iter` | `int` | `1000` | Maximum iterations per run. |
| `init_temperature` | `float` | `1000` | Initial temperature. |
| `stopping_temperature` | `float` | `1e-08` | Temperature stopping criterion. |
| `cooling_rate` | `float` | `0.99` | Geometric cooling factor ($0 < \alpha < 1$). |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Simulated Annealing                      |
| Acronym           | SA                                       |
| Year Introduced   | 1983                                     |
| Authors           | Kirkpatrick, Scott; Gelatt, C. Daniel; Vecchi, Mario |
| Algorithm Class   | Classical                                |
| Complexity        | $O(\text{iterations} \times \text{evaluations})$              |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Acceptance probability (Metropolis criterion):

$$
P(\text{accept}) = \begin{cases}
1 & \text{if } \Delta E < 0 \\
e^{-\Delta E / T} & \text{if } \Delta E \geq 0
\end{cases}
$$

where:
- $\Delta E = E(x_{new}) - E(x_{current})$ is energy (fitness) change
- $T$ is the current temperature
- Cooling schedule: $T_{k+1} = \alpha \cdot T_k$ (geometric cooling)

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Reject out-of-bounds solutions

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| init_temperature       | 100.0   | 10-1000          | Initial temperature            |
| stopping_temperature   | 1e-8    | 1e-10            | Stopping criterion             |
| cooling_rate           | 0.99    | 0.95-0.999       | Temperature reduction factor   |
| max_iter               | 1000    | 10000            | Maximum iterations per run     |
| population_size        | 100     | 10-50            | Number of restarts             |

**Sensitivity Analysis**:
- `cooling_rate`: **High** impact (slower=better exploration, faster=faster convergence)
- `init_temperature`: **Medium** impact on early exploration
- Recommended: $\alpha \in [0.95, 0.999]$, $T_0 \in [10, 1000]$

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
- Time per iteration: $O(1)$ per proposal
- Space complexity: $O(n)$
- BBOB budget usage: _30-70% of $\text{dim} \times 10000$_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, Rugged landscapes
- **Weak function classes**: Highly smooth (slower than gradient methods)
- Success rate at 1e-8: **40-70%** (dim=5, multimodal)

**Convergence Properties**:
- Convergence rate: Probabilistic, depends on cooling schedule
- Local vs Global: Can escape local optima (probabilistic acceptance)
- Premature convergence risk: **Low** (if cooling slow enough)

**Reproducibility**:
- **Deterministic**: Yes (given same seed)
- **BBOB compliance**: seed required for 15 runs
- RNG: `numpy.random.default_rng(self.seed)`

**Known Limitations**:
- Cooling schedule critical to performance
- Slow convergence compared to gradient methods on smooth functions
- No convergence guarantees for arbitrary schedules

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: COCO/BBOB compliance

## References

[1] Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by simulated annealing."
_Science_, 220(4598), 671-680.
https://doi.org/10.1126/science.220.4598.671

[2] Metropolis, N., et al. (1953). "Equation of state calculations by fast computing machines."
_The Journal of Chemical Physics_, 21(6), 1087-1092.
https://doi.org/10.1063/1.1699114

[3] Hansen, N., Auger, A., et al. (2021). "COCO: A platform for comparing continuous optimizers."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

## See Also

HillClimbing: Greedy local search without probabilistic acceptance
BBOB Comparison: SA better on multimodal, HC faster on unimodal
TabuSearch: Memory-based metaheuristic
BBOB Comparison: Both escape local optima, different mechanisms

## Benchmark Performance

Interactive fitness landscape of a representative multimodal benchmark function (drag to rotate, scroll to zoom):

<ClientOnly>
  <FitnessLandscape3D functionName="rastrigin" />
</ClientOnly>

Convergence, final-fitness distribution and performance profile on `rastrigin` (5D), averaged over independent runs (compared against representative baselines):

<ClientOnly>
  <BenchmarkCharts
    algorithm="SimulatedAnnealing"
    functionName="rastrigin"
    :dimension="5"
    :compareWith="['GreyWolfOptimizer', 'ParticleSwarm', 'AntColony']"
  />
</ClientOnly>

## Related Pages

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`simulated_annealing.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/simulated_annealing.py)
:::
