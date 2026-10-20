# Gradient-Based API

API reference for the 11 classes of `opt.gradient_based`. This page is generated from the source; see the [algorithm pages](/algorithms/gradient-based/) for formulations and references.

## Imports

```python
from opt.gradient_based import (
    AdamW,
    AdaMax,
    AdaDelta,
    ADAGrad,
    ADAMOptimization,
    AMSGrad,
    NesterovAcceleratedGradient,
    Nadam,
    RMSprop,
    SGDMomentum,
    SGD,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`AdamW`](/algorithms/gradient-based/adamw) | Adam with Decoupled Weight Decay | `learning_rate=ADAMW_LEARNING_RATE`, `beta1=ADAM_BETA1`, `beta2=ADAM_BETA2`, `epsilon=ADAM_EPSILON`, `weight_decay=ADAMW_WEIGHT_DECAY` |
| [`AdaMax`](/algorithms/gradient-based/adamax) | Adamax | `learning_rate=0.002`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-08` |
| [`AdaDelta`](/algorithms/gradient-based/adadelta) | Adaptive Delta | `rho=0.97`, `eps=1e-08` |
| [`ADAGrad`](/algorithms/gradient-based/adagrad) | Adaptive Gradient Algorithm | `lr=0.01`, `eps=1e-08` |
| [`ADAMOptimization`](/algorithms/gradient-based/adaptive-moment-estimation) | Adaptive Moment Estimation | `alpha=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-13` |
| [`AMSGrad`](/algorithms/gradient-based/amsgrad) | AMSGrad | `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-08` |
| [`NesterovAcceleratedGradient`](/algorithms/gradient-based/nesterov-accelerated-gradient) | Nesterov Accelerated Gradient | `learning_rate=0.01`, `momentum=0.9` |
| [`Nadam`](/algorithms/gradient-based/nadam) | Nesterov-accelerated Adaptive Moment | `learning_rate=NADAM_LEARNING_RATE`, `beta1=ADAM_BETA1`, `beta2=ADAM_BETA2`, `epsilon=ADAM_EPSILON` |
| [`RMSprop`](/algorithms/gradient-based/rmsprop) | Root Mean Square Propagation | `learning_rate=0.01`, `rho=0.9`, `epsilon=1e-08` |
| [`SGDMomentum`](/algorithms/gradient-based/sgd-momentum) | SGD with Momentum | `learning_rate=0.01`, `momentum=0.9` |
| [`SGD`](/algorithms/gradient-based/stochastic-gradient-descent) | Stochastic Gradient Descent | `learning_rate=0.01` |
