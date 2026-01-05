# AMSGrad

<span class="badge badge-gradient">Gradient-Based</span>

AMSGrad Optimizer.

## Algorithm Overview

This module implements the AMSGrad optimization algorithm. AMSGrad is a variant of Adam
that fixes the exponential moving average issue in Adam. It ensures that the second moment
estimate never decreases, which helps with convergence to the optimal solution.

AMSGrad performs the following update rule:
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient^2
    v_hat = max(v_hat, v)
    m_hat = m / (1 - beta1^t)
    v_hat_corrected = v_hat / (1 - beta2^t)
    x = x - learning_rate * m_hat / (sqrt(v_hat_corrected) + epsilon)

where:
    - x: current solution
    - m: first moment estimate (exponential moving average of gradients)
    - v: second moment estimate (exponential moving average of squared gradients)
    - v_hat: maximum of all v up to current time step
    - learning_rate: step size for parameter updates
    - beta1, beta2: exponential decay rates for moment estimates
    - epsilon: small constant for numerical stability
    - t: time step

## Usage

```python
from opt.gradient_based.amsgrad import AMSGrad
from opt.benchmark.functions import sphere

optimizer = AMSGrad(
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
| `learning_rate` | `float` | `0.001` | Learning rate (step size). |
| `beta1` | `float` | `0.9` | Exponential decay rate for first moment estimates. |
| `beta2` | `float` | `0.999` | Exponential decay rate for second moment estimates. |
| `epsilon` | `float` | `1e-08` | Small constant for numerical stability. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`amsgrad.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/amsgrad.py)
:::
