# RMSprop

<span class="badge badge-gradient">Gradient-Based</span>

RMSprop Optimizer.

## Algorithm Overview

This module implements the RMSprop optimization algorithm. RMSprop is an adaptive
learning rate method that was proposed by Geoffrey Hinton. It modifies AdaGrad to
perform better in non-convex settings by using a moving average of squared gradients
instead of accumulating all squared gradients.

RMSprop performs the following update rule:
    v = rho * v + (1 - rho) * gradient^2
    x = x - (learning_rate / sqrt(v + epsilon)) * gradient

where:
    - x: current solution
    - v: moving average of squared gradients
    - learning_rate: step size for parameter updates
    - rho: decay rate (typically 0.9)
    - epsilon: small constant to avoid division by zero
    - gradient: gradient of the objective function at x

## Usage

```python
from opt.gradient_based.rmsprop import RMSprop
from opt.benchmark.functions import sphere

optimizer = RMSprop(
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
| `rho` | `float` | `0.9` | The decay rate for the moving average. |
| `epsilon` | `float` | `1e-08` | Small constant to avoid division by zero. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`rmsprop.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/rmsprop.py)
:::
