# Gravitational Search Optimizer

<span class="badge badge-physics">Physics-Inspired</span>

Gravitational Search Algorithm (GSA).

## Algorithm Overview

This module implements the Gravitational Search Algorithm, a physics-inspired
metaheuristic based on Newton's law of gravity and laws of motion.

Objects (solutions) attract each other with gravitational forces proportional
to their mass (fitness) and inversely proportional to distance. Heavier masses
(better solutions) attract lighter masses (worse solutions).

## Reference

> Rashedi, E., Nezamabadi-Pour, H., & Saryazdi, S. (2009). GSA: A Gravitational Search Algorithm. Information Sciences, 179(13), 2232-2248. DOI: 10.1016/j.ins.2009.03.004

[📄 View Paper (DOI: 10.1016/j.ins.2009.03.004)](https://doi.org/10.1016/j.ins.2009.03.004)

## Usage

```python
from opt.physics_inspired.gravitational_search import GravitationalSearchOptimizer
from opt.benchmark.functions import sphere

optimizer = GravitationalSearchOptimizer(
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
| `g0` | `float` | `_GRAVITATIONAL_CONSTANT_INITIAL` | Algorithm-specific parameter |
| `alpha` | `float` | `_GRAVITATIONAL_DECAY_RATE` | Step size or learning rate |

## See Also

- [Physics-Inspired Algorithms](/algorithms/physics-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`gravitational_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/physics_inspired/gravitational_search.py)
:::
