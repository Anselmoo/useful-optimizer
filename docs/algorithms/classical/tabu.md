# Tabu Search

<span class="badge badge-classical">Classical</span>

Tabu Search metaheuristic optimization algorithm.

## Algorithm Overview

This module implements the Tabu Search optimization algorithm.

The Tabu Search algorithm is a metaheuristic optimization algorithm that is used to
solve combinatorial optimization problems. It is inspired by the concept of memory in
human search behavior. The algorithm maintains a tabu list that keeps track of recently
visited solutions and prevents the search from revisiting them in the near future. This
helps the algorithm to explore different regions of the search space and avoid getting
stuck in local optima.

This module provides the `TabuSearch` class, which is an implementation of the
Tabu Search algorithm. It can be used to minimize a given objective function
over a continuous search space.

## Usage

```python
from opt.classical.tabu_search import TabuSearch
from opt.benchmark.functions import sphere

optimizer = TabuSearch(
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
| `tabu_list_size` | `int` | `50` | Maximum size of tabu memory. |
| `neighborhood_size` | `int` | `10` | Number of neighbors evaluated per iteration. |
| `seed` | `int  \|  None` | `None` | Random seed for BBOB reproducibility. |

## Algorithm Metadata

| Property          | Value                                    |
|-------------------|------------------------------------------|
| Algorithm Name    | Tabu Search                              |
| Acronym           | TS                                       |
| Year Introduced   | 1986                                     |
| Authors           | Glover, Fred                             |
| Algorithm Class   | Classical                                |
| Complexity        | $O(\text{population} \times \text{neighbors} \times \text{iterations})$   |
| Properties        | Derivative-free, Stochastic          |
| Implementation    | Python 3.10+                             |
| COCO Compatible   | Yes                                      |

## Mathematical Formulation

Neighborhood exploration with tabu memory:

$$
x_{t+1} = \arg\min_{x' \in N(x_t) \setminus T} f(x')
$$

where:
- $N(x_t)$ is the neighborhood of current solution
- $T$ is the tabu list (forbidden recent moves)
- Aspiration criterion: accept tabu move if $f(x') < f(x^*)$ (best so far)

Tabu list update:
- Add selected move to tabu list
- Remove oldest move if list exceeds `tabu_list_size`

Constraint handling:
- **Boundary conditions**: Clamping to bounds
- **Feasibility enforcement**: Natural during neighborhood generation

## Hyperparameters

| Parameter              | Default | BBOB Recommended | Description                    |
|------------------------|---------|------------------|--------------------------------|
| population_size        | 100     | 10-50            | Number of independent runs     |
| max_iter               | 1000    | 5000-10000       | Maximum iterations per run     |
| tabu_list_size         | 50      | dim to $5 \times \text{dim}$     | Tabu memory size               |
| neighborhood_size      | 10      | 10-20            | Neighbors evaluated per iter   |

**Sensitivity Analysis**:
- `tabu_list_size`: **High** impact (too small=cycling, too large=restricted search)
- `neighborhood_size`: **Medium** impact on exploration quality
- Recommended: $|T| \in [\text{dim}, 5 \times \text{dim}]$

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
- Time per iteration: $O(|N| \times |T|)$ for neighborhood and tabu checks
- Space complexity: $O(|T| + |P|)$ for tabu list and population
- BBOB budget usage: _40-70% of $\text{dim} \times 10000$_

**BBOB Performance Characteristics**:
- **Best function classes**: Multimodal, Discrete-like landscapes
- **Weak function classes**: Smooth unimodal (slower than gradient methods)
- Success rate at 1e-8: **35-60%** (dim=5, multimodal)

**Convergence Properties**:
- Convergence rate: Depends on tabu list size and neighborhood
- Local vs Global: Escapes local optima via tabu memory
- Premature convergence risk: **Medium** (tabu list prevents revisiting)

**Reproducibility**:
- **Deterministic**: Yes (given same seed)
- **BBOB compliance**: seed required for 15 runs
- RNG: `numpy.random.default_rng(self.seed)`

**Known Limitations**:
- Tabu list size critical (too small=cycling, too large=restricted)
- Neighborhood generation strategy affects performance
- No convergence guarantees for arbitrary tabu strategies

**Version History**:
- v0.1.0: Initial implementation
- v0.1.2: COCO/BBOB compliance

## References

[1] Glover, F. (1986). "Future paths for integer programming and links to artificial intelligence."
_Computers & Operations Research_, 13(5), 533-549.
https://doi.org/10.1016/0305-0548(86)90048-1

[2] Glover, F., & Laguna, M. (1997). "Tabu Search."
_Kluwer Academic Publishers_, Boston.

[3] Hansen, N., Auger, A., et al. (2021). "COCO: A platform for comparing continuous optimizers."
_Optimization Methods and Software_, 36(1), 114-144.
https://doi.org/10.1080/10556788.2020.1808977

**COCO Data Archive**:
- Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
- Code repository: https://github.com/Anselmoo/useful-optimizer

## See Also

SimulatedAnnealing: Probabilistic metaheuristic without memory
BBOB Comparison: Both escape local optima, TS uses deterministic memory
HillClimbing: Greedy local search without memory or probabilistic acceptance
BBOB Comparison: TS better on multimodal due to tabu memory

## Related Pages

- [Classical Algorithms](/algorithms/classical/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`tabu_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/tabu_search.py)
:::
