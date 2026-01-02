# AdamW

<span class="badge badge-gradient">Gradient-Based</span>

AdamW Optimizer.

## Algorithm Overview

This module implements the AdamW optimization algorithm. AdamW is a variant of Adam
that decouples weight decay from the gradient-based update. This decoupling provides
better regularization and often leads to improved generalization in machine learning.

AdamW performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    x = x - learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * x)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - v: second moment estimate (exponential moving average of squared gradients)
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - epsilon: small constant for numerical stability
    - weight_decay: weight decay coefficient
    - t: time step

## Usage

```python
from opt.gradient_based.adamw import AdamW
from opt.benchmark.functions import sphere

optimizer = AdamW(
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
| `max_iter` | `int` | `DEFAULT_MAX_ITERATIONS` | Maximum iterations. |
| `learning_rate` | `float` | `ADAMW_LEARNING_RATE` | Learning rate (step size). |
| `beta1` | `float` | `ADAM_BETA1` | Exponential decay rate for first moment estimates. |
| `beta2` | `float` | `ADAM_BETA2` | Exponential decay rate for second moment estimates. |
| `epsilon` | `float` | `ADAM_EPSILON` | Small constant for numerical stability. |
| `weight_decay` | `float` | `ADAMW_WEIGHT_DECAY` | Weight decay coefficient for L2 regularization decoupled from gradient. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `target_precision` | `float` | `1e-08` | Algorithm-specific parameter |
| `f_opt` | `float  \|  None` | `None` | Algorithm-specific parameter |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adamw.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/adamw.py)
:::
