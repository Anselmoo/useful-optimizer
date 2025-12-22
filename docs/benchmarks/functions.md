# Benchmark Functions

This page documents the benchmark functions available in Useful Optimizer for testing and comparing optimization algorithms.

## Overview

All benchmark functions are located in `opt.benchmark.functions` and share the same interface:

```python
from opt.benchmark.functions import sphere, rosenbrock, ackley

# Each function takes a numpy array and returns a scalar
import numpy as np
x = np.array([1.0, 2.0, 3.0])
fitness = sphere(x)  # Returns: 14.0
```

## Unimodal Functions

### Sphere Function

The simplest test function - a sum of squares.

$$
f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2
$$

| Property | Value |
|----------|-------|
| **Optimum** | $f(\mathbf{0}) = 0$ |
| **Bounds** | $[-5.12, 5.12]^n$ |
| **Modality** | Unimodal |
| **Separability** | Separable |

```python
from opt.benchmark.functions import sphere
import numpy as np

x = np.zeros(10)
print(sphere(x))  # 0.0
```

### Rosenbrock Function

A classic ill-conditioned function with a narrow valley.

$$
f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
$$

| Property | Value |
|----------|-------|
| **Optimum** | $f(\mathbf{1}) = 0$ |
| **Bounds** | $[-5, 10]^n$ |
| **Modality** | Unimodal |
| **Separability** | Non-separable |

```python
from opt.benchmark.functions import rosenbrock
import numpy as np

x = np.ones(10)
print(rosenbrock(x))  # 0.0
```

## Multi-Modal Functions

### Rastrigin Function

Highly multi-modal with regular local minima distribution.

$$
f(\mathbf{x}) = 10n + \sum_{i=1}^{n} \left[ x_i^2 - 10\cos(2\pi x_i) \right]
$$

| Property | Value |
|----------|-------|
| **Optimum** | $f(\mathbf{0}) = 0$ |
| **Bounds** | $[-5.12, 5.12]^n$ |
| **Modality** | Multi-modal (~$10^n$ local minima) |
| **Separability** | Separable |

```python
from opt.benchmark.functions import rastrigin
import numpy as np

x = np.zeros(10)
print(rastrigin(x))  # 0.0
```

### Ackley Function

Multi-modal with a nearly flat outer region and central funnel.

$$
f(\mathbf{x}) = -20\exp\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right) + 20 + e
$$

| Property | Value |
|----------|-------|
| **Optimum** | $f(\mathbf{0}) = 0$ |
| **Bounds** | $[-32.768, 32.768]^n$ |
| **Modality** | Multi-modal |
| **Separability** | Non-separable |

```python
from opt.benchmark.functions import ackley
import numpy as np

x = np.zeros(10)
print(ackley(x))  # ≈ 0.0 (4.44e-16)
```

### Shifted Ackley Function

A shifted version of Ackley with non-centered optimum.

```python
from opt.benchmark.functions import shifted_ackley
import numpy as np

# Optimum is shifted from origin
x = np.array([1.0, 1.0])  # Example shift
print(shifted_ackley(x))
```

### Griewank Function

Many regularly distributed local minima but increasingly flat in higher dimensions.

$$
f(\mathbf{x}) = \sum_{i=1}^{n} \frac{x_i^2}{4000} - \prod_{i=1}^{n} \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1
$$

| Property | Value |
|----------|-------|
| **Optimum** | $f(\mathbf{0}) = 0$ |
| **Bounds** | $[-600, 600]^n$ |
| **Modality** | Multi-modal |
| **Separability** | Non-separable |

```python
from opt.benchmark.functions import griewank
import numpy as np

x = np.zeros(10)
print(griewank(x))  # 0.0
```

## Function Characteristics Summary

| Function | Unimodal | Separable | Difficulty |
|----------|----------|-----------|------------|
| Sphere | ✅ | ✅ | Easy |
| Rosenbrock | ✅ | ❌ | Medium |
| Rastrigin | ❌ | ✅ | Hard |
| Ackley | ❌ | ❌ | Hard |
| Griewank | ❌ | ❌ | Medium |

## Usage in Benchmarks

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import (
    sphere,
    rosenbrock,
    rastrigin,
    ackley,
    griewank
)

functions = {
    'sphere': {'func': sphere, 'bounds': (-5.12, 5.12)},
    'rosenbrock': {'func': rosenbrock, 'bounds': (-5, 10)},
    'rastrigin': {'func': rastrigin, 'bounds': (-5.12, 5.12)},
    'ackley': {'func': ackley, 'bounds': (-32.768, 32.768)},
    'griewank': {'func': griewank, 'bounds': (-600, 600)}
}

for name, config in functions.items():
    optimizer = ParticleSwarm(
        func=config['func'],
        lower_bound=config['bounds'][0],
        upper_bound=config['bounds'][1],
        dim=10,
        max_iter=100
    )
    _, fitness = optimizer.search()
    print(f"{name}: {fitness:.6e}")
```

## Creating Custom Functions

You can create custom objective functions for your specific problems:

```python
import numpy as np
from opt.swarm_intelligence import ParticleSwarm

def my_function(x: np.ndarray) -> float:
    """Custom objective function.

    Args:
        x: Input vector of shape (n,)

    Returns:
        Scalar fitness value (lower is better)
    """
    return np.sum(x**2) + 10 * np.sin(np.sum(x))

optimizer = ParticleSwarm(
    func=my_function,
    lower_bound=-10.0,
    upper_bound=10.0,
    dim=10,
    max_iter=100
)

best_solution, best_fitness = optimizer.search()
```

## References

1. Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions for global optimization problems. *IJMMNO*, 4(2), 150-194.

2. Suganthan, P. N., et al. (2005). Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization. *KanGAL Report*, 2005.
