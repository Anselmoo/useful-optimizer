---
title: Algorithm Comparison
description: Performance comparison across optimization algorithms
---

# Algorithm Comparison

This page provides performance comparisons between different optimization algorithms on standard benchmark functions.

---

## Methodology

Each algorithm was tested with:

- **Dimensions:** 2, 10, 30
- **Max iterations:** 1000
- **Population size:** 30 (where applicable)
- **Runs per configuration:** 30 (for statistical significance)

---

## Results Summary

### Performance on Sphere Function

| Algorithm | 2D | 10D | 30D |
|-----------|-----|-----|-----|
| **BFGS** | ✓✓✓ | ✓✓✓ | ✓✓ |
| **Adam** | ✓✓✓ | ✓✓ | ✓✓ |
| **PSO** | ✓✓ | ✓✓ | ✓ |
| **DE** | ✓✓✓ | ✓✓✓ | ✓✓ |
| **CMA-ES** | ✓✓✓ | ✓✓✓ | ✓✓✓ |

Legend: ✓✓✓ Excellent, ✓✓ Good, ✓ Acceptable

### Performance on Rosenbrock Function

| Algorithm | 2D | 10D | 30D |
|-----------|-----|-----|-----|
| **BFGS** | ✓✓✓ | ✓✓ | ✓ |
| **Adam** | ✓✓ | ✓ | ✓ |
| **PSO** | ✓✓ | ✓ | - |
| **DE** | ✓✓✓ | ✓✓ | ✓✓ |
| **CMA-ES** | ✓✓✓ | ✓✓✓ | ✓✓ |

### Performance on Ackley Function

| Algorithm | 2D | 10D | 30D |
|-----------|-----|-----|-----|
| **BFGS** | ✓ | - | - |
| **Adam** | ✓ | - | - |
| **PSO** | ✓✓✓ | ✓✓ | ✓ |
| **DE** | ✓✓✓ | ✓✓✓ | ✓✓ |
| **CMA-ES** | ✓✓✓ | ✓✓✓ | ✓✓✓ |

---

## Recommendations

Based on comprehensive testing:

### For Smooth Functions

!!! tip "Recommendation"
    Use **gradient-based methods** (BFGS, Adam) for smooth, differentiable functions. They converge faster when gradients are available.

### For Multimodal Functions

!!! tip "Recommendation"
    Use **population-based methods** (PSO, DE, CMA-ES) for functions with many local minima. They're better at global exploration.

### For High Dimensions

!!! tip "Recommendation"
    **CMA-ES** and **Differential Evolution** scale best to high dimensions (30+). Avoid methods that don't handle curse of dimensionality well.

---

## Running Your Own Comparisons

```python
from opt.benchmark.functions import sphere, rosenbrock, shifted_ackley
from opt.swarm_intelligence import ParticleSwarm
from opt.evolutionary import DifferentialEvolution
from opt.classical import BFGS
import numpy as np

def compare_algorithms(func, dim, n_runs=10):
    """Compare multiple algorithms on a benchmark function."""
    algorithms = {
        "PSO": lambda: ParticleSwarm(func=func, lower_bound=-5, upper_bound=5, dim=dim, max_iter=100),
        "DE": lambda: DifferentialEvolution(func=func, lower_bound=-5, upper_bound=5, dim=dim, max_iter=100),
        "BFGS": lambda: BFGS(func=func, lower_bound=-5, upper_bound=5, dim=dim),
    }

    results = {}
    for name, create_opt in algorithms.items():
        fitnesses = []
        for _ in range(n_runs):
            opt = create_opt()
            _, fitness = opt.search()
            fitnesses.append(fitness)
        results[name] = {
            "mean": np.mean(fitnesses),
            "std": np.std(fitnesses),
            "best": np.min(fitnesses),
        }
        print(f"{name}: mean={results[name]['mean']:.2e} ± {results[name]['std']:.2e}")

    return results

# Run comparison
print("=== Sphere Function (10D) ===")
compare_algorithms(sphere, dim=10)
```
