---
title: Ant Colony Optimization
description: Nature-inspired algorithm based on ant foraging behavior
icon: material/bug
tags:
  - swarm-intelligence
  - metaheuristic
  - population-based
---

# Ant Colony Optimization

<span class="algorithm-badge algorithm-badge--swarm">Swarm Intelligence</span>

!!! abstract "Algorithm Summary"
    **Category:** Swarm Intelligence
    **Year:** 1992
    **Author:** Marco Dorigo
    **Complexity:** \(O(n \cdot m \cdot t)\) where \(n\) = dimensions, \(m\) = ants, \(t\) = iterations

---

## Overview

Ant Colony Optimization (ACO) is inspired by the foraging behavior of ants finding optimal paths between their colony and food sources. Ants deposit pheromones on paths they travel, and other ants are more likely to follow paths with stronger pheromone concentrations, leading to emergent path optimization.

---

## Mathematical Formulation

The probability of ant \(k\) selecting component \(j\) at step \(i\) is:

\[
p_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}
\]

Pheromone update rule:

\[
\tau_{ij}(t+1) = (1 - \rho) \cdot \tau_{ij}(t) + \sum_{k=1}^{m} \Delta\tau_{ij}^k
\]

Where:

| Symbol | Description |
|--------|-------------|
| \(\tau_{ij}\) | Pheromone level on edge \((i,j)\) |
| \(\eta_{ij}\) | Heuristic desirability of edge \((i,j)\) |
| \(\alpha\) | Pheromone influence exponent |
| \(\beta\) | Heuristic influence exponent |
| \(\rho\) | Pheromone evaporation rate |
| \(\mathcal{N}_i^k\) | Feasible neighborhood for ant \(k\) at \(i\) |

---

## Usage

```python
from opt.swarm_intelligence import AntColony
from opt.benchmark.functions import shifted_ackley

optimizer = AntColony(
    func=shifted_ackley,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    population_size=20,
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
| `population_size` | `int` | `20` | Number of ants |
| `max_iter` | `int` | `100` | Maximum iterations |
| `evaporation_rate` | `float` | `0.5` | Pheromone evaporation rate \(\rho\) |
| `alpha` | `float` | `1.0` | Pheromone influence |
| `beta` | `float` | `2.0` | Heuristic influence |

---

## Advantages & Limitations

!!! success "Advantages"
    - Excellent for combinatorial optimization
    - Robust to changes in the environment
    - Distributed computing friendly
    - Positive feedback leads to rapid convergence

!!! warning "Limitations"
    - Primarily designed for discrete problems
    - Convergence can be slow
    - Many parameters to tune
    - May converge to local optima

---

## References

!!! quote "Original Paper"
    Dorigo, M. (1992). *Optimization, Learning and Natural Algorithms*.
    PhD Thesis, Politecnico di Milano.

!!! quote "Foundational Paper"
    Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: A cooperative learning approach to the traveling salesman problem.
    *IEEE Transactions on Evolutionary Computation*, 1(1), 53-66.
    [DOI: 10.1109/4235.585892](https://doi.org/10.1109/4235.585892)
