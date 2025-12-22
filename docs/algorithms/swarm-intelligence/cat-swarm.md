# Cat Swarm Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Cat Swarm Optimization (CSO) algorithm.

## Algorithm Overview

This module implements the Cat Swarm Optimization (CSO) algorithm, which is a
population-based optimization algorithm inspired by the behavior of cats. The algorithm
aims to find the optimal solution for a given optimization problem by simulating the
hunting behavior of cats.

The CSO algorithm is implemented in the `CatSwarmOptimization` class, which inherits
from the `AbstractOptimizer` class. The `CatSwarmOptimization` class provides methods
to initialize the population, perform seeking mode and tracing mode operations, and run
the CSO algorithm to find the optimal solution.

Example usage:
    optimizer = CatSwarmOptimization(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        cats=100,
        max_iter=2000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}")

## Usage

```python
from opt.swarm_intelligence.cat_swarm_optimization import CatSwarmOptimization
from opt.benchmark.functions import sphere

optimizer = CatSwarmOptimization(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | Required | The objective function to be minimized. |
| `dim` | `int` | Required | The dimensionality of the problem. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `cats` | `int` | `50` | The number of cats in the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `seeking_memory_pool` | `int` | `5` | The size of the seeking memory pool. |
| `counts_of_dimension_to_change` | `int  \|  None` | `None` | The number of dimensions to change during seeking mode. |
| `smp_change_probability` | `float` | `0.1` | The probability of changing dimensions during seeking mode. |
| `spc_probability` | `float` | `0.2` | The probability of performing tracing mode. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cat_swarm_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/cat_swarm_optimization.py)
:::
