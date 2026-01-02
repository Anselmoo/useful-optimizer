# Particle Swarm Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Particle Swarm Optimization (PSO) algorithm implementation.

## Algorithm Overview

This module provides an implementation of the Particle Swarm Optimization (PSO) algorithm for solving optimization problems.
PSO is a population-based stochastic optimization algorithm inspired by the social behavior of bird flocking or fish schooling.

The main class in this module is `ParticleSwarm`, which represents the PSO algorithm. It takes an objective function, lower and upper bounds of the search space, dimensionality of the search space, and other optional parameters as input. The `search` method performs the PSO optimization and returns the best solution found.

Example usage:
    optimizer = ParticleSwarm(
        func=shifted_ackley,
        lower_bound=-32.768,
        upper_bound=+32.768,
        dim=2,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Classes:
    - ParticleSwarm: Particle Swarm Optimization (PSO) algorithm for optimization problems.

## Usage

```python
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.benchmark.functions import sphere

optimizer = ParticleSwarm(
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
| `population_size` | `int` | `DEFAULT_POPULATION_SIZE` | Number of particles in swarm. |
| `max_iter` | `int` | `DEFAULT_MAX_ITERATIONS` | Maximum iterations. |
| `c1` | `float` | `PSO_COGNITIVE_COEFFICIENT` | Cognitive coefficient controlling attraction to personal
        best. |
| `c2` | `float` | `PSO_SOCIAL_COEFFICIENT` | Social coefficient controlling attraction to global best. |
| `w` | `float` | `PSO_INERTIA_WEIGHT` | Inertia weight controlling previous velocity influence. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `track_history` | `bool` | `False` | Enable convergence history tracking for BBOB
        post-processing. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`particle_swarm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/particle_swarm.py)
:::
