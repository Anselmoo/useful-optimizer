# Adadelta

<span class="badge badge-gradient">Gradient-Based</span>

AdaDelta Optimizer.

## Algorithm Overview

This module implements the AdaDelta optimizer, which is an extension of AdaGrad that
seeks to reduce its sensitivity to the learning rate hyperparameter.

AdaDelta is a gradient-based optimization algorithm that adapts the learning rate
for each of the parameters in the model. It is designed to converge faster than
AdaGrad by using a moving average of the squared gradient values to scale the learning rate.

The AdaDelta optimizer is defined by the following update rule:

    Eg = rho * Eg + (1 - rho) * g^2
    dx = -sqrt(Edx + eps) / sqrt(Eg + eps) * g
    Edx = rho * Edx + (1 - rho) * dx^2
    x = x + dx

where:
    - x: current solution
    - g: gradient of the objective function
    - rho: decay rate
    - eps: small constant to avoid dividing by zero
    - Eg: moving average of squared gradient values
    - Edx: moving average of squared updates

The algorithm iteratively updates the solution x by computing the gradient of the
objective function at x, scaling it by the moving average of the squared gradients,
and dividing it by the square root of the moving average of the squared updates.

The algorithm continues for a fixed number of iterations or until a specified
stopping criterion is met, returning the best solution found.

This module provides a simple example of how to use the AdaDelta optimizer to minimize
the Shifted Ackley's function in two dimensions.

## Usage

```python
from opt.gradient_based.adadelta import AdaDelta
from opt.benchmark.functions import sphere

optimizer = AdaDelta(
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
| `rho` | `float` | `0.97` | Decay rate for moving averages of squared gradients and updates. |
| `eps` | `float` | `1e-08` | Small constant for numerical stability in division operations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adadelta.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/adadelta.py)
:::
