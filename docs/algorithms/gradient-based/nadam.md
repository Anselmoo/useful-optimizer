# NAdam

<span class="badge badge-gradient">Gradient-Based</span>

Nadam Optimizer.

## Algorithm Overview

This module implements the Nadam (Nesterov-accelerated Adaptive Moment Estimation)
optimization algorithm. Nadam combines Adam with Nesterov momentum, incorporating
lookahead into the gradient computation which can lead to faster convergence.

Nadam performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    m_bar = beta1 * m_hat + (1 - beta1) * gradient / (1 - beta1^t)
    x = x - learning_rate * m_bar / (sqrt(v_hat) + epsilon)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - v: second moment estimate (exponential moving average of squared gradients)
    - m_bar: Nesterov-corrected first moment estimate
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - epsilon: small constant for numerical stability
    - t: time step

## Usage

```python
from opt.gradient_based.nadam import Nadam
from opt.benchmark.functions import sphere

optimizer = Nadam(
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
| `max_iter` | `int` | `DEFAULT_MAX_ITERATIONS` | The maximum number of iterations. |
| `learning_rate` | `float` | `NADAM_LEARNING_RATE` | The learning rate. |
| `beta1` | `float` | `ADAM_BETA1` | Exponential decay rate for first moment estimates. |
| `beta2` | `float` | `ADAM_BETA2` | Exponential decay rate for second moment estimates. |
| `epsilon` | `float` | `ADAM_EPSILON` | Small constant for numerical stability. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`nadam.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/nadam.py)
:::
