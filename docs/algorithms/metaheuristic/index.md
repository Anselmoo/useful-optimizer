---
title: Metaheuristic Algorithms
description: High-level algorithmic frameworks for optimization
tags:
  - metaheuristic
---

# Metaheuristic Algorithms

Metaheuristics are high-level problem-independent algorithmic frameworks that provide strategies for exploring search spaces effectively.

---

## Overview

These algorithms are characterized by:

- **Problem independence** - Applicable to various problem types
- **Balance** - Exploration vs exploitation trade-off
- **Stochasticity** - Random components for diversity
- **Flexibility** - Easy to adapt and hybridize

---

## Algorithms

| Algorithm | Inspiration | Description |
|-----------|-------------|-------------|
| **Harmony Search** | Music | Musical improvisation |
| **Cross Entropy** | Statistics | Importance sampling |
| **Eagle Strategy** | Nature | Two-phase search |
| **Particle Filter** | Statistics | Sequential Monte Carlo |
| **Sine Cosine** | Mathematics | Trigonometric oscillation |
| **Shuffled Frog Leaping** | Nature | Memetic frogs |
| **Stochastic Diffusion** | Physics | Diffusion processes |
| **Stochastic Fractal** | Mathematics | Fractal patterns |
| **Variable Depth** | Graph Theory | Variable-depth search |
| **VNS** | Neighborhoods | Systematic neighborhood change |
| **VLSNS** | Neighborhoods | Very large neighborhoods |
| **Colliding Bodies** | Physics | Collision dynamics |

---

## Usage Pattern

```python
from opt.metaheuristic import HarmonySearch
from opt.benchmark.functions import sphere

optimizer = HarmonySearch(
    func=sphere,
    lower_bound=-10,
    upper_bound=10,
    dim=5,
    population_size=20,
    max_iter=100,
)

best_solution, best_fitness = optimizer.search()
```

---

## When to Use

!!! success "Good For"
    - Black-box optimization
    - Complex search landscapes
    - When simple methods fail
    - Hybrid algorithm development

!!! warning "Limitations"
    - May require parameter tuning
    - Can be slower than specialized methods
    - Performance varies by problem
