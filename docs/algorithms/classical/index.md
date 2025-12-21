---
title: Classical Methods
description: Traditional optimization methods and local search techniques
tags:
  - classical
  - local-search
---

# Classical Optimization Methods

Classical methods include well-established mathematical optimization techniques and local search algorithms developed over decades of research.

---

## Overview

These algorithms represent the foundation of optimization theory:

- **Derivative-based** - BFGS, Conjugate Gradient, Trust Region
- **Derivative-free** - Nelder-Mead, Powell's Method
- **Local search** - Hill Climbing, Simulated Annealing, Tabu Search

---

## Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **BFGS** | Quasi-Newton | Approximates Hessian inverse |
| **L-BFGS** | Quasi-Newton | Memory-efficient BFGS |
| **Conjugate Gradient** | Gradient | Conjugate directions |
| **Trust Region** | Model-based | Trusted step bounds |
| **Nelder-Mead** | Simplex | Derivative-free simplex |
| **Powell** | Direction Set | Derivative-free directions |
| **Hill Climbing** | Local Search | Greedy improvement |
| **Simulated Annealing** | Probabilistic | Temperature-based acceptance |
| **Tabu Search** | Memory-based | Short-term memory |

---

## Usage Pattern

```python
from opt.classical import BFGS
from opt.benchmark.functions import rosenbrock

optimizer = BFGS(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=5,
    dim=2,
    num_restarts=5,
)

best_solution, best_fitness = optimizer.search()
```

---

## When to Use

!!! success "Good For"
    - Smooth, convex functions (BFGS, CG)
    - Derivative-free problems (Nelder-Mead, Powell)
    - Local refinement of solutions
    - Escaping local optima (SA)

!!! warning "Limitations"
    - Many converge to local optima
    - May require multiple restarts
    - Some need gradient information
