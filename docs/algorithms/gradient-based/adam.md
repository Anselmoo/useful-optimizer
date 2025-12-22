# Adam

<span class="badge badge-gradient">Gradient-Based</span>

Adaptive Moment Estimation (Adam) Optimizer.

## Algorithm Overview

This module implements the Adam optimization algorithm. Adam is a gradient-based
optimization algorithm that computes adaptive learning rates for each parameter. It
combines the advantages of two other extensions of stochastic gradient descent:

    - AdaGrad
    - RMSProp

Adam works well in practice and compares favorably to other adaptive learning-method
algorithms as it converges fast and the learning speed of the Model is quite fast and
efficient. It is straightforward to implement, is computationally efficient, has little
memory requirements, is invariant to diagonal rescaling of the gradients, and is well
suited for problems that are large in terms of data and/or parameters.

## Usage

```python
from opt.gradient_based.adaptive_moment_estimation import ADAMOptimization
from opt.benchmark.functions import sphere

optimizer = ADAMOptimization(
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
| `alpha` | `float` | `0.001` | The learning rate. |
| `beta1` | `float` | `0.9` | The exponential decay rate for the first moment estimates. |
| `beta2` | `float` | `0.999` | The exponential decay rate for the second moment estimates. |
| `epsilon` | `float` | `1e-13` | A small value added to the denominator for numerical stability. |
| `seed` | `int  \|  None` | `None` | The random seed for reproducibility. |

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`adaptive_moment_estimation.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/gradient_based/adaptive_moment_estimation.py)
:::
