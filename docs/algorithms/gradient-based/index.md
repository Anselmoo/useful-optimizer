# Gradient-Based Algorithms

First-order and second-order optimization algorithms that use gradient information.

## Overview

Gradient-based optimizers use derivative information to guide the search toward optima. These methods are particularly effective for smooth, differentiable functions and are the foundation of modern deep learning.

## Available Algorithms

### First-Order Methods

- [SGD with Momentum](./sgd-momentum) - Stochastic Gradient Descent with momentum
- [Adam](./adam) - Adaptive Moment Estimation
- [AdamW](./adamw) - Adam with weight decay
- [RMSprop](./rmsprop) - Root Mean Square Propagation
- [Adagrad](./adagrad) - Adaptive Gradient Algorithm
- [Adadelta](./adadelta) - Extension of Adagrad
- [Nadam](./nadam) - Nesterov-accelerated Adam
- [AMSGrad](./amsgrad) - Adam variant with long-term memory

## Usage Example

```python
from opt.gradient_based import Adam, SGDMomentum
from opt.benchmark.functions import rosenbrock

# Adam optimizer
adam = Adam(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=10,
    dim=10,
    max_iter=1000,
    learning_rate=0.01
)
best_solution, best_fitness = adam.search()
```

## See Also

- [API Reference](/api/gradient-based) - Complete API documentation
- [Classical Methods](../classical/) - For quasi-Newton methods like BFGS
