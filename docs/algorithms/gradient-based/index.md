# 🧠 Gradient-Based Algorithms

Optimization using gradient information.

This category contains **11 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Adam with Decoupled Weight Decay](./adamw) | ADAMW | 2017 | Adam with Decoupled Weight Decay (AdamW) optimization algorithm. |
| [Adamax](./adamax) | ADAMAX | 2014 | Adamax optimization algorithm. |
| [Adaptive Delta](./adadelta) | ADADELTA | 2012 | Adaptive Delta (AdaDelta) optimization algorithm. |
| [Adaptive Gradient Algorithm](./adagrad) | ADAGRAD | 2011 | Adaptive Gradient Algorithm (AdaGrad) optimization algorithm. |
| [Adaptive Moment Estimation](./adaptive-moment-estimation) | ADAM | 2014 | Adaptive Moment Estimation (Adam) optimization algorithm. |
| [AMSGrad](./amsgrad) | AMSGRAD | 2018 | AMSGrad optimization algorithm. |
| [Nesterov Accelerated Gradient](./nesterov-accelerated-gradient) | NAG | 1983 | Nesterov Accelerated Gradient (NAG) optimization algorithm. |
| [Nesterov-accelerated Adaptive Moment](./nadam) | NADAM | 2016 | Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm. |
| [Root Mean Square Propagation](./rmsprop) | RMSPROP | 2012 | Root Mean Square Propagation (RMSprop) optimization algorithm. |
| [SGD with Momentum](./sgd-momentum) | SGD-M | 1964 | Stochastic Gradient Descent with Momentum (SGD-M) optimization algorithm. |
| [Stochastic Gradient Descent](./stochastic-gradient-descent) | SGD | 1951 | Stochastic Gradient Descent (SGD) optimization algorithm. |

## Usage

Every optimizer shares the same interface:

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
```

## See Also

- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)
