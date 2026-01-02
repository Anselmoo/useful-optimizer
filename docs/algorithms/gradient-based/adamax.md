# Adamax

<span class="badge badge-gradient">Gradient-Based</span>

AdaMax Optimizer.

## Algorithm Overview

This module implements the AdaMax optimization algorithm. AdaMax is a variant of Adam
that uses the infinity norm instead of the L2 norm for the second moment estimate.
This makes it less sensitive to outliers in gradients and can be more stable in some cases.

AdaMax performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    u = max(beta2 * u, |gradient|)
    x = x - (learning_rate / (1 - beta1^t)) * (m / u)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - u: second moment estimate (exponential moving average of infinity norm of gradients)
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - t: time step

## Usage

```python
from opt.gradient_based.adamax import AdaMax
from opt.benchmark.functions import sphere

optimizer = AdaMax(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
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
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `learning_rate` | `float` | `0.002` | Learning rate (step size). |
| `beta1` | `float` | `0.9` | Exponential decay rate for first moment estimates. |
| `beta2` | `float` | `0.999` | Exponential decay rate for infinity norm. |
| `epsilon` | `float` | `1e-08` | Small constant for numerical stability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adamax.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/adamax.py)
:::
