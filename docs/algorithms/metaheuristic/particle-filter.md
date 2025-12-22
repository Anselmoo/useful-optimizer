# Particle Filter

<span class="badge badge-metaheuristic">Metaheuristic</span>

Particle Filter Algorithm.

## Algorithm Overview

This module implements the Particle Filter algorithm. Particle filters, or Sequential
Monte Carlo (SMC) methods, are a set of on-line posterior density estimation algorithms
that estimate the posterior density of the state-space by directly implementing the
Bayesian recursion equations.

The main idea behind particle filters is to represent the posterior density function by
a set of random samples, or particles, and assign a weight to each particle that
represents the probability of that particle being sampled from the probability density
function.

Particle filters are particularly useful for non-linear and non-Gaussian estimation
problems.

## Usage

```python
from opt.metaheuristic.particle_filter import ParticleFilter
from opt.benchmark.functions import sphere

optimizer = ParticleFilter(
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
| `population_size` | `int` | `100` | Number of individuals in population |
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `inertia` | `float` | `0.7` | Algorithm-specific parameter |
| `cognitive` | `float` | `1.5` | Algorithm-specific parameter |
| `social` | `float` | `1.5` | Algorithm-specific parameter |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`particle_filter.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/particle_filter.py)
:::
