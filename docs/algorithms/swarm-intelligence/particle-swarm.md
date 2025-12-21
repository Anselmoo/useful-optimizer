---
title: Particle Swarm Optimization
description: Social-inspired metaheuristic based on bird flocking behavior
icon: material/bird
status: new
tags:
  - swarm-intelligence
  - metaheuristic
  - population-based
---

# Particle Swarm Optimization

<span class="algorithm-badge algorithm-badge--swarm">Swarm Intelligence</span>

!!! abstract "Algorithm Summary"
    **Category:** Swarm Intelligence
    **Year:** 1995
    **Authors:** Kennedy & Eberhart
    **Complexity:** \(O(n \cdot p \cdot t)\) where \(n\) = dimensions, \(p\) = population, \(t\) = iterations

---

## Overview

Particle Swarm Optimization (PSO) is a population-based stochastic optimization algorithm inspired by the social behavior of bird flocking or fish schooling. Each particle represents a candidate solution that moves through the search space, influenced by its own best-known position and the swarm's best-known position.

---

## Mathematical Formulation

The velocity and position updates are defined as:

\[
v_i^{t+1} = \omega v_i^t + c_1 r_1 (p_i - x_i^t) + c_2 r_2 (g - x_i^t)
\]

\[
x_i^{t+1} = x_i^t + v_i^{t+1}
\]

Where:

| Symbol | Description |
|--------|-------------|
| \(v_i\) | Velocity of particle \(i\) |
| \(x_i\) | Position of particle \(i\) |
| \(\omega\) | Inertia weight (controls exploration vs exploitation) |
| \(c_1\) | Cognitive coefficient (personal influence) |
| \(c_2\) | Social coefficient (swarm influence) |
| \(r_1, r_2\) | Random values in \([0, 1]\) |
| \(p_i\) | Personal best position of particle \(i\) |
| \(g\) | Global best position found by the swarm |

---

## Algorithm Pseudocode

```
Initialize swarm with random positions and velocities
Evaluate fitness of all particles
Set personal best (pbest) for each particle
Set global best (gbest) as best pbest

while not converged:
    for each particle i:
        Update velocity using formula
        Update position using formula
        Evaluate fitness
        if fitness < pbest_fitness[i]:
            pbest[i] = position[i]
        if fitness < gbest_fitness:
            gbest = position[i]

return gbest, gbest_fitness
```

---

## Usage

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-2.768,
    upper_bound=2.768,
    dim=2,
    population_size=30,
    max_iter=100,
)
best_solution, best_fitness = optimizer.search()

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | *required* | Objective function to minimize |
| `lower_bound` | `float` | *required* | Lower boundary of search space |
| `upper_bound` | `float` | *required* | Upper boundary of search space |
| `dim` | `int` | *required* | Number of dimensions |
| `population_size` | `int` | `30` | Number of particles in the swarm |
| `max_iter` | `int` | `100` | Maximum number of iterations |
| `w` | `float` | `0.7` | Inertia weight |
| `c1` | `float` | `1.5` | Cognitive (personal) coefficient |
| `c2` | `float` | `1.5` | Social (global) coefficient |

### Parameter Guidelines

!!! tip "Tuning Recommendations"

    - **Inertia weight (\(\omega\))**: Start with 0.7-0.9. Higher values encourage exploration.
    - **Cognitive coefficient (\(c_1\))**: Usually 1.5-2.0. Higher values make particles trust their own experience more.
    - **Social coefficient (\(c_2\))**: Usually 1.5-2.0. Higher values make particles follow the swarm.
    - **Population size**: 20-50 for most problems. Increase for complex multimodal functions.

---

## Variants

PSO has many variants that improve performance:

| Variant | Description |
|---------|-------------|
| **Standard PSO** | Original formulation |
| **Constriction PSO** | Uses constriction factor for stability |
| **Adaptive PSO** | Dynamically adjusts parameters |
| **Bare-bones PSO** | Simplified velocity update |

---

## Advantages & Limitations

!!! success "Advantages"
    - Simple concept and easy implementation
    - Few parameters to tune
    - No gradient information required
    - Good global search capability
    - Parallelizable

!!! warning "Limitations"
    - May converge prematurely on complex landscapes
    - Performance depends on parameter settings
    - Not ideal for highly constrained problems
    - May struggle with high-dimensional spaces (>100D)

---

## When to Use

PSO is well-suited for:

- **Continuous optimization** problems
- **Black-box optimization** where gradients are unavailable
- **Multimodal functions** where exploration matters
- Problems with **moderate dimensionality** (2-100D)

---

## References

!!! quote "Original Paper"
    Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    *Proceedings of ICNN'95 - International Conference on Neural Networks*, 4, 1942-1948.
    [DOI: 10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968)

!!! quote "Review Paper"
    Poli, R., Kennedy, J., & Blackwell, T. (2007). Particle swarm optimization: An overview.
    *Swarm Intelligence*, 1(1), 33-57.
    [DOI: 10.1007/s11721-007-0002-0](https://doi.org/10.1007/s11721-007-0002-0)
