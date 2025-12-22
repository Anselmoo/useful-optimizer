# Nesterov Accelerated Gradient

<span class="badge badge-gradient">Gradient-Based</span>

Nesterov Accelerated Gradient (NAG) Optimizer.

## Algorithm Overview

This module implements the Nesterov Accelerated Gradient optimization algorithm. NAG is
an improvement over SGD with Momentum that provides better convergence rates. The key
idea is to compute the gradient not at the current position, but at an approximate
future position, which provides better gradient information.

NAG performs the following update rule:
    v = momentum * v - learning_rate * gradient(x + momentum * v)
    x = x + v

where:
    - x: current solution
    - v: velocity (momentum term)
    - learning_rate: step size for parameter updates
    - momentum: momentum coefficient (typically 0.9)
    - gradient: gradient of the objective function

## Usage

```python
from opt.gradient_based.nesterov_accelerated_gradient import NesterovAcceleratedGradient
from opt.benchmark.functions import sphere

optimizer = NesterovAcceleratedGradient(
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
| `func` | `Callable` | Required | The objective function to be optimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `learning_rate` | `float` | `0.01` | The learning rate. |
| `momentum` | `float` | `0.9` | The momentum coefficient. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`nesterov_accelerated_gradient.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/nesterov_accelerated_gradient.py)
:::
