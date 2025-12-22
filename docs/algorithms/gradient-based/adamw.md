# AdamW Optimizer

<span class="badge badge-gradient">Gradient-Based</span>

AdamW is an improved variant of Adam that decouples weight decay from the gradient update, leading to better regularization.

## Algorithm Overview

AdamW was introduced by Loshchilov and Hutter (2017) as a fix to Adam's weight decay implementation. It's widely used in deep learning and continuous optimization.

## Mathematical Formulation

### Gradient Moments

First moment (mean):
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

Second moment (uncentered variance):
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

### Bias Correction

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

### Parameter Update

$$
\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

Where $\lambda$ is the weight decay coefficient.

## Usage

```python
from opt.gradient_based import AdamW
from opt.benchmark.functions import rosenbrock

optimizer = AdamW(
    func=rosenbrock,
    lower_bound=-5.0,
    upper_bound=10.0,
    dim=10,
    max_iter=1000,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01
)

best_solution, best_fitness = optimizer.search()
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | Required | Objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `max_iter` | `int` | 1000 | Maximum iterations |
| `learning_rate` | `float` | 0.001 | Step size α |
| `beta1` | `float` | 0.9 | First moment decay rate |
| `beta2` | `float` | 0.999 | Second moment decay rate |
| `epsilon` | `float` | 1e-8 | Numerical stability term |
| `weight_decay` | `float` | 0.01 | L2 regularization coefficient |

## AdamW vs. Adam

The key difference is in how weight decay is applied:

| Aspect | Adam | AdamW |
|--------|------|-------|
| Weight decay | Added to gradient | Applied after update |
| L2 regularization | Coupled with momentum | Decoupled |
| Effective regularization | Inconsistent | Consistent |

### Why Decoupling Matters

In Adam, weight decay is scaled by the adaptive learning rate, leading to:
- Smaller weight decay for parameters with large gradients
- Inconsistent regularization across parameters

AdamW fixes this by applying weight decay directly to the weights.

## When to Use AdamW

::: tip Recommended For
- Smooth, differentiable objective functions
- Training neural networks
- Problems where regularization is important
- Convex and near-convex landscapes
:::

::: warning Limitations
- Requires differentiable objective
- May get stuck in local optima
- Not suitable for non-smooth or discrete problems
- Requires tuning of learning rate
:::

## Hyperparameter Guidelines

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| Learning rate | 1e-4 to 1e-2 | Start with 1e-3 |
| β₁ | 0.9 | Rarely needs tuning |
| β₂ | 0.999 | Rarely needs tuning |
| Weight decay | 1e-4 to 1e-1 | Higher for stronger regularization |

## References

1. Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv:1711.05101*.

2. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.

## See Also

- [Adam](./adam) - Original Adam optimizer
- [SGD Momentum](./sgd-momentum) - Simpler gradient descent
- [RMSprop](./rmsprop) - Adaptive learning rate method
