# Adagrad

<span class="badge badge-gradient">Gradient-Based</span>

ADAGrad Optimizer.

## Algorithm Overview

This module implements the Adaptive Gradient Algorithm (ADAGrad) optimizer. ADAGrad is
a gradient-based optimization algorithm that adapts the learning rate to the parameters,
performing smaller updates for parameters associated with frequently occurring features,
and larger updates for parameters associated with infrequent features. It is particularly
useful for dealing with sparse data.

ADAGrad's main strength is that it eliminates the need to manually tune the learning rate.
Most implementations also include a 'smoothing term' to avoid division by zero when the
gradient is zero.

The ADAGrad optimizer is commonly used in machine learning and deep learning applications.

## Usage

```python
from opt.gradient_based.adagrad import ADAGrad
from opt.benchmark.functions import sphere

optimizer = ADAGrad(
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
| `lr` | `float` | `0.01` | The learning rate. |
| `eps` | `float` | `1e-08` | A small value added to the denominator for numerical stability. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adagrad.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/adagrad.py)
:::
