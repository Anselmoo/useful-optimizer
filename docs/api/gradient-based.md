# Gradient-Based Algorithms API

API reference for gradient-based optimizers in `opt.gradient_based`.

## Module Overview

```python
from opt.gradient_based import (
    SGDMomentum,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    Nadam,
    AMSGrad,
)
```

## Common Interface

```python
class GradientOptimizer(AbstractOptimizer):
    def __init__(
        self,
        func: Callable,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        learning_rate: float = 0.01,
        **kwargs
    ):
        pass

    def search(self) -> tuple[np.ndarray, float]:
        pass
```

## Available Algorithms

- `SGDMomentum` - SGD with momentum
- `Adam` - Adaptive Moment Estimation
- `AdamW` - Adam with weight decay
- `RMSprop` - Root Mean Square Propagation
- `Adagrad` - Adaptive Gradient
- `Adadelta` - Extension of Adagrad
- `Nadam` - Nesterov-accelerated Adam
- `AMSGrad` - Adam with long-term memory

## Example Usage

```python
from opt.gradient_based import Adam, RMSprop
from opt.benchmark.functions import sphere

# Adam optimizer
adam = Adam(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=1000,
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999
)
solution, fitness = adam.search()

# RMSprop
rmsprop = RMSprop(
    func=sphere,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=1000,
    learning_rate=0.001,
    decay_rate=0.9
)
solution, fitness = rmsprop.search()
```

## See Also

- [Gradient-Based Algorithms](/algorithms/gradient-based/) - Algorithm details
