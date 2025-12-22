# Salp Swarm Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Salp Swarm Algorithm (SSA).

## Algorithm Overview

This module implements the Salp Swarm Algorithm, a nature-inspired metaheuristic
based on the swarming behavior of salps in oceans.

Salps form chains to move effectively through water. The leader at the front
navigates, while followers chain together behind. This behavior is modeled
mathematically for optimization.

## Reference

> Mirjalili, S., Gandomi, A. H., Mirjalili, S. Z., Saremi, S., Faris, H., & Mirjalili, S. M. (2017). Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems. Advances in Engineering Software, 114, 163-191. DOI: 10.1016/j.advengsoft.2017.07.002

[📄 View Paper (DOI: 10.1016/j.advengsoft.2017.07.002)](https://doi.org/10.1016/j.advengsoft.2017.07.002)

## Usage

```python
from opt.swarm_intelligence.salp_swarm_algorithm import SalpSwarmOptimizer
from opt.benchmark.functions import sphere

optimizer = SalpSwarmOptimizer(
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
| `func` | `Callable` | Required | Objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |
| `population_size` | `int` | `100` | Number of individuals in population |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`salp_swarm_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/salp_swarm_algorithm.py)
:::
