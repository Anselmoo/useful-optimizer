# Stochastic Gradient Descent

<span class="badge badge-gradient">Gradient-Based</span>

Stochastic Gradient Descent (SGD) Optimizer.

## Algorithm Overview

This module implements the Stochastic Gradient Descent optimization algorithm. SGD is
a gradient-based optimization algorithm that updates parameters in the direction
opposite to the gradient of the objective function. It is one of the most fundamental
and widely-used optimization algorithms in machine learning.

SGD performs the following update rule:
    x = x - learning_rate * gradient

where:
    - x: current solution
    - learning_rate: step size for parameter updates
    - gradient: gradient of the objective function at x

## Usage

```python
from opt.gradient_based.stochastic_gradient_descent import SGD
from opt.benchmark.functions import sphere

optimizer = SGD(
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
| `learning_rate` | `float` | `0.01` | Learning rate (step size). |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`stochastic_gradient_descent.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/stochastic_gradient_descent.py)
:::
