---
title: Evolutionary Algorithms
description: Population-based algorithms using principles of natural evolution
tags:
  - evolutionary
  - population-based
---

# Evolutionary Algorithms

Evolutionary algorithms are population-based optimization methods inspired by biological evolution, using mechanisms such as selection, mutation, crossover, and survival of the fittest.

---

## Overview

These algorithms maintain a population of candidate solutions that evolve over generations. Key principles include:

- **Selection** - Fitter individuals are more likely to reproduce
- **Variation** - New solutions created through mutation and recombination
- **Inheritance** - Offspring inherit traits from parents
- **Competition** - Solutions compete for survival

---

## Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **CMA-ES** | Covariance matrix adaptation | High-dimensional, ill-conditioned |
| **Cultural Algorithm** | Cultural evolution model | Knowledge-guided search |
| **Differential Evolution** | Vector differences for mutation | Continuous optimization |
| **EDA** | Probabilistic model building | Structure learning |
| **Genetic Algorithm** | Natural selection simulation | Discrete/combinatorial |
| **Imperialist Competitive** | Empire competition model | Multi-population search |

---

## Usage Pattern

```python
from opt.evolutionary import DifferentialEvolution
from opt.benchmark.functions import rosenbrock

optimizer = DifferentialEvolution(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,
    max_iter=200,
)

best_solution, best_fitness = optimizer.search()
```

---

## When to Use

!!! success "Good For"
    - High-dimensional continuous optimization
    - Multimodal landscapes
    - Problems without gradient information
    - Global optimization

!!! warning "Limitations"
    - Can be computationally expensive
    - Many parameters to tune
    - May require large populations
