---
title: Test Functions
description: Standard benchmark functions for optimization testing
---

# Test Functions

Benchmark functions are mathematical functions with known properties used to evaluate and compare optimization algorithms.

---

## Available Functions

### Unimodal Functions

These functions have a single global minimum, useful for testing convergence speed.

#### Sphere Function

The simplest benchmark function, a sum of squares:

\[
f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2
\]

**Properties:**

- **Global minimum:** \(f(\mathbf{0}) = 0\)
- **Search domain:** \([-5.12, 5.12]^n\)
- **Characteristics:** Convex, separable, unimodal

```python
from opt.benchmark.functions import sphere
import numpy as np

x = np.array([0.0, 0.0])
print(f"sphere([0, 0]) = {sphere(x)}")  # Output: 0.0
```

---

#### Rosenbrock Function

A classic non-convex function with a narrow curved valley:

\[
f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
\]

**Properties:**

- **Global minimum:** \(f(\mathbf{1}) = 0\)
- **Search domain:** \([-5, 10]^n\)
- **Characteristics:** Non-convex, non-separable, unimodal

```python
from opt.benchmark.functions import rosenbrock
import numpy as np

x = np.array([1.0, 1.0])
print(f"rosenbrock([1, 1]) = {rosenbrock(x)}")  # Output: 0.0
```

---

### Multimodal Functions

These functions have many local minima, testing global search capability.

#### Ackley Function

A widely-used multimodal test function:

\[
f(\mathbf{x}) = -20 \exp\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right) + 20 + e
\]

**Properties:**

- **Global minimum:** \(f(\mathbf{0}) = 0\)
- **Search domain:** \([-32.768, 32.768]^n\)
- **Characteristics:** Nearly flat outer region, deep hole at center

```python
from opt.benchmark.functions import ackley, shifted_ackley
import numpy as np

x = np.array([0.0, 0.0])
print(f"ackley([0, 0]) = {ackley(x):.6f}")
```

!!! note "Shifted Ackley"
    `shifted_ackley` is a variant where the global minimum is not at the origin, making it harder to "cheat" by always starting at zero.

---

#### Rastrigin Function

A highly multimodal function with regularly distributed local minima:

\[
f(\mathbf{x}) = 10n + \sum_{i=1}^{n} \left[ x_i^2 - 10\cos(2\pi x_i) \right]
\]

**Properties:**

- **Global minimum:** \(f(\mathbf{0}) = 0\)
- **Search domain:** \([-5.12, 5.12]^n\)
- **Characteristics:** Large number of local minima (\(10^n\))

```python
from opt.benchmark.functions import rastrigin
import numpy as np

x = np.array([0.0, 0.0])
print(f"rastrigin([0, 0]) = {rastrigin(x):.6f}")
```

---

## Function Summary Table

| Function | Type | Minimum | Domain | Difficulty |
|----------|------|---------|--------|------------|
| `sphere` | Unimodal | \(f(\mathbf{0}) = 0\) | \([-5.12, 5.12]^n\) | Easy |
| `rosenbrock` | Unimodal | \(f(\mathbf{1}) = 0\) | \([-5, 10]^n\) | Medium |
| `ackley` | Multimodal | \(f(\mathbf{0}) = 0\) | \([-32.768, 32.768]^n\) | Medium |
| `shifted_ackley` | Multimodal | Shifted | \([-32.768, 32.768]^n\) | Hard |
| `rastrigin` | Multimodal | \(f(\mathbf{0}) = 0\) | \([-5.12, 5.12]^n\) | Hard |

---

## Usage Example

```python
from opt.benchmark.functions import sphere, rosenbrock, shifted_ackley
from opt.swarm_intelligence import ParticleSwarm

functions = [
    ("Sphere", sphere, -5, 5),
    ("Rosenbrock", rosenbrock, -5, 5),
    ("Shifted Ackley", shifted_ackley, -2.768, 2.768),
]

for name, func, lb, ub in functions:
    optimizer = ParticleSwarm(
        func=func,
        lower_bound=lb,
        upper_bound=ub,
        dim=5,
        population_size=30,
        max_iter=100,
    )
    solution, fitness = optimizer.search()
    print(f"{name}: fitness = {fitness:.6e}")
```

---

## Custom Functions

You can define your own objective functions:

```python
import numpy as np

def custom_function(x):
    """Custom optimization target."""
    return np.sum(x**2) + np.prod(np.abs(x))

from opt.classical import BFGS

optimizer = BFGS(
    func=custom_function,
    lower_bound=-10,
    upper_bound=10,
    dim=5,
)
solution, fitness = optimizer.search()
```

!!! tip "Function Requirements"
    Your objective function should:

    - Accept a NumPy array as input
    - Return a single scalar value
    - Be defined over the specified bounds
