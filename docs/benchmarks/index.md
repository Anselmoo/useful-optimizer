---
title: Benchmarks
description: Test functions and algorithm comparisons
---

# Benchmarks

Useful Optimizer includes a comprehensive suite of benchmark functions for testing and comparing optimization algorithms.

---

## Overview

<div class="grid cards" markdown>

-   :material-function:{ .lg .middle } **Test Functions**

    ---

    Collection of standard benchmark functions for optimization.

    [:octicons-arrow-right-24: View Functions](functions.md)

-   :material-chart-bar:{ .lg .middle } **Algorithm Comparison**

    ---

    Performance comparisons across different algorithms.

    [:octicons-arrow-right-24: View Comparisons](comparison.md)

</div>

---

## Quick Start

```python
from opt.benchmark.functions import (
    sphere,
    rosenbrock,
    shifted_ackley,
    rastrigin,
)

# Test any optimizer on benchmark functions
from opt.swarm_intelligence import ParticleSwarm

for func in [sphere, rosenbrock, shifted_ackley]:
    optimizer = ParticleSwarm(
        func=func,
        lower_bound=-10,
        upper_bound=10,
        dim=5,
        max_iter=100,
    )
    _, fitness = optimizer.search()
    print(f"{func.__name__}: {fitness:.6f}")
```
