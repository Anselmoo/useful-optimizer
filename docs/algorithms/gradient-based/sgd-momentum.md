# SGD with Momentum

<span class="badge badge-gradient">Gradient-Based</span>

Stochastic Gradient Descent with Momentum Optimizer.

## Algorithm Overview

This module implements the SGD with Momentum optimization algorithm. SGD with Momentum
is an extension of SGD that accelerates gradient descent in the relevant direction and
dampens oscillations. It does this by adding a fraction of the update vector of the
past time step to the current update vector.

SGD with Momentum performs the following update rule:
    v = momentum * v - learning_rate * gradient
    x = x + v

where:
    - x: current solution
    - v: velocity (momentum term)
    - learning_rate: step size for parameter updates
    - momentum: momentum coefficient (typically 0.9)
    - gradient: gradient of the objective function at x

## Usage

```python
from opt.gradient_based.sgd_momentum import SGDMomentum
from opt.benchmark.functions import sphere

optimizer = SGDMomentum(
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
| `momentum` | `float` | `0.9` | Momentum coefficient. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`sgd_momentum.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/sgd_momentum.py)
:::
