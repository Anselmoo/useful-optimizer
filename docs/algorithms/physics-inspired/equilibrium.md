# Equilibrium Optimizer

<span class="badge badge-physics">Physics-Inspired</span>

Equilibrium Optimizer (EO).

## Algorithm Overview

This module implements the Equilibrium Optimizer, a physics-inspired metaheuristic
based on control volume mass balance models used to estimate dynamic and equilibrium
states.

The algorithm uses concepts from mass balance to describe concentration changes
in a control volume, simulating particles reaching equilibrium states.

## Reference

> Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020). Equilibrium optimizer: A novel optimization algorithm. Knowledge-Based Systems, 191, 105190. DOI: 10.1016/j.knosys.2019.105190

[📄 View Paper (DOI: 10.1016/j.knosys.2019.105190)](https://doi.org/10.1016/j.knosys.2019.105190)

## Usage

```python
from opt.physics_inspired.equilibrium_optimizer import EquilibriumOptimizer
from opt.benchmark.functions import sphere

optimizer = EquilibriumOptimizer(
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
| `a1` | `float` | `_A1` | Algorithm-specific parameter |
| `a2` | `float` | `_A2` | Algorithm-specific parameter |
| `gp` | `float` | `_GP` | Algorithm-specific parameter |

## See Also

- [Physics-Inspired Algorithms](/algorithms/physics-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`equilibrium_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/physics_inspired/equilibrium_optimizer.py)
:::
