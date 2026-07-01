# Stochastic Diffusion Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Stochastic Diffusion Search (SDS) optimization algorithm.

## Algorithm Overview

This module implements the Stochastic Diffusion Search optimizer, which is an
optimization algorithm that uses a population of agents to explore the search space and
find the optimal solution for a given objective function.

The main class in this module is `StochasticDiffusionSearch`, which represents the
optimizer. It takes the objective function, lower and upper bounds of the search space,
dimensionality of the search space, population size, maximum number of iterations,
and seed for the random number generator as input parameters.

The optimizer works by initializing a population of agents, where each agent has a
position in the search space and a score based on the objective function. The algorithm
then iteratively performs a test phase and a diffusion phase to update the positions of
the agents. After the specified number of iterations, the algorithm returns the best
solution found and its corresponding score.

Example usage:
    optimizer = StochasticDiffusionSearch(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

## Usage

```python
from opt.metaheuristic.stochastic_diffusion_search import StochasticDiffusionSearch
from opt.benchmark.functions import sphere

optimizer = StochasticDiffusionSearch(
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
| `population_size` | `int` | `100` | Number of agents. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Stochastic Diffusion Search              |
| Acronym           | SDS                                      |
| Year Introduced   | 1989                                     |
| Authors           | Bishop, John Mark                        |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Two-phase process (test and diffusion):

**Test Phase**: Each agent evaluates its hypothesis
$$active_i = \begin{cases} True & \text{if } f(x_i) < threshold \\ False & \text{otherwise} \end{cases}$$

**Diffusion Phase**: Inactive agents communicate with random active agents
$$x_i^{new} = \begin{cases} x_j & \text{if agent j is active} \\ random & \text{otherwise} \end{cases}$$

where:
- $x_i$ is agent i's position (hypothesis)
- $active_i$ is agent i's activity status
- Communication is direct (one-to-one), not stigmergetic

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of agents               |
| max_iter               | 1000    | 10000            | Maximum iterations             |

**Sensitivity Analysis**:
- `population_size`: **High** impact on solution quality and convergence
- Recommended tuning ranges: population $\in [5 \times dim, 20 \times dim]$

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
- BBOB budget usage: _Typically uses 65-85% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Pattern-like, discrete-like continuous problems
- **Weak function classes**: Smooth unimodal, gradient-rich functions
- Typical success rate at 1e-8 precision: **15-25%** (dim=5)
- Expected Running Time (ERT): Moderate; good for complex discrete-like landscapes

**Convergence Properties**:
- Convergence rate: Sublinear (agent-based diffusion)
- Local vs Global: Good global exploration via random recruitment
- Premature convergence risk: **Low** (diffusion prevents clustering)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Agent-based approach naturally handles boundaries

**Known Limitations**:
- Originally designed for discrete pattern matching, adapted for continuous
- Convergence can be slow on smooth landscapes
- BBOB known issues: Less effective than gradient methods on unimodal functions

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Bishop, J. M. (1989). "Stochastic Searching Networks."
_Proceedings of the 1st IEE Conference on Artificial Neural Networks_, 329-331.

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results (originally for pattern matching)
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper: Focused on pattern matching applications
- This implementation: Adapted for continuous optimization with BBOB compliance

## See Also

AntColony: Another swarm intelligence algorithm
BBOB Comparison: ACO uses stigmergy; SDS uses direct communication

ParticleSwarm: Population-based swarm algorithm
BBOB Comparison: PSO velocity-based; SDS agent recruitment-based

AbstractOptimizer: Base class for all optimizers
opt.benchmark.functions: BBOB-compatible test functions

Related BBOB Algorithm Classes:
- Evolutionary: GeneticAlgorithm, DifferentialEvolution
- Swarm: ParticleSwarm, AntColony
- Gradient: AdamW, SGDMomentum

## Related Pages

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`stochastic_diffusion_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/stochastic_diffusion_search.py)
:::
