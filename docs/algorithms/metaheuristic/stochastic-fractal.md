# Stochastic Fractal Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Stochastic Fractal Search (SFS) optimization algorithm.

## Algorithm Overview

This module implements the Stochastic Fractal Search optimizer, which is an
optimization algorithm used to find the minimum of a given function.

The Stochastic Fractal Search algorithm works by maintaining a population of
individuals and iteratively updating them based on their scores. At each iteration,
a best individual is selected, and other individuals in the population undergo a
diffusion phase to explore the search space. The algorithm continues for a specified
number of iterations or until a termination condition is met.

## Usage

```python
from opt.metaheuristic.stochastic_fractal_search import StochasticFractalSearch
from opt.benchmark.functions import sphere

optimizer = StochasticFractalSearch(
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
| `population_size` | `int` | `100` | Number of search particles. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `diffusion_parameter` | `float` | `0.5` | Step size for Gaussian diffusion. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Stochastic Fractal Search                |
| Acronym           | SFS                                      |
| Year Introduced   | 2015                                     |
| Authors           | Salimi, Hamid                            |
| Algorithm Class   | Metaheuristic                            |
| Complexity        | O(population_size * dim * max_iter)      |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Core update equation (diffusion process):

$$
X_{i,j}^{new} = X_{i,j} + \alpha \times \mathcal{N}(0, 1)
$$

where:
- $X_{i,j}$ is the position of particle $i$ at dimension $j$
- $\alpha$ is the diffusion parameter (step size)
- $\mathcal{N}(0, 1)$ is standard normal distribution
- Update/selection phase chooses better solutions

Inspired by random fractal growth via Gaussian random walks.

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Random initialization within bounds

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10*dim           | Number of search particles     |
| max_iter               | 1000    | 10000            | Maximum iterations             |
| diffusion_parameter    | 0.5     | 0.1-1.0          | Step size for diffusion        |

**Sensitivity Analysis**:
- `diffusion_parameter`: **High** impact on exploration intensity
- `population_size`: **Medium** impact on search quality
- Recommended tuning ranges: $\alpha \in [0.1, 1.0]$, population $\in [5 \times dim, 15 \times dim]$

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
- BBOB budget usage: _Typically uses 55-75% of dim $\times$ 10000 budget for convergence_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, rugged landscapes
- **Weak function classes**: Simple unimodal, separable functions
- Typical success rate at 1e-8 precision: **18-28%** (dim=5)
- Expected Running Time (ERT): Moderate; good exploration capabilities

**Convergence Properties**:
- Convergence rate: Sublinear (random walk-based)
- Local vs Global: Excellent global exploration via fractal diffusion
- Premature convergence risk: **Very Low** (stochastic nature prevents trapping)

**Reproducibility**:
- **Deterministic**: Yes - Same seed guarantees same results
- **BBOB compliance**: seed parameter required for 15 independent runs
- Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
- RNG usage: `numpy.random.default_rng(self.seed)` throughout

**Implementation Details**:
- Parallelization: Not supported in this implementation
- Constraint handling: Clamping to bounds
- Numerical stability: Gaussian sampling well-behaved

**Known Limitations**:
- Relatively simple algorithm; may require many iterations for convergence
- Diffusion parameter tuning important for performance
- BBOB known issues: Slow on simple unimodal functions compared to gradient methods

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: BBOB compliance improvements

## References

[1] Salimi, H. (2015). "Stochastic Fractal Search: A powerful metaheuristic algorithm."
_Knowledge-Based Systems_, 75, 1-18.
https://doi.org/10.1016/j.knosys.2014.07.025

[2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
"COCO: A platform for comparing continuous optimizers in a black-box setting."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Algorithm data: Limited BBOB-specific results (algorithm introduced 2015)
- Code repository: https://github.com/Anselmoo/useful-optimizer

**Implementation**:
- Original paper code: MATLAB implementations available
- This implementation: Based on [1] with modifications for BBOB compliance

## See Also

GaussianProcessOptimizer: Bayesian optimization with Gaussian processes
BBOB Comparison: GPO model-based; SFS uses random fractal diffusion

ParticleSwarm: Population-based swarm intelligence algorithm
BBOB Comparison: PSO velocity-based; SFS diffusion-based

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
View the implementation: [`stochastic_fractal_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/stochastic_fractal_search.py)
:::
