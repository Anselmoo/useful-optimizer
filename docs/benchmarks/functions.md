# Benchmark Functions

A comprehensive collection of benchmark functions for testing optimization algorithms.

## Overview

Useful Optimizer includes 20+ benchmark functions covering:

- Unimodal functions
- Multimodal functions
- Functions with plateaus
- Non-separable functions

## Function Catalog

### Sphere Function

The simplest benchmark - a smooth, convex, unimodal function.

$$
f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2
$$

| Property | Value |
|----------|-------|
| **Global Minimum** | $f(\mathbf{0}) = 0$ |
| **Search Domain** | $x_i \in [-5.12, 5.12]$ |
| **Dimensions** | Any |
| **Modality** | Unimodal |

```python
from opt.benchmark.functions import sphere
import numpy as np

x = np.array([0, 0])
print(f"f([0, 0]) = {sphere(x)}")  # 0.0
```

---

### Rosenbrock Function

A classic non-convex function with a narrow curved valley.

$$
f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1-x_i)^2 \right]
$$

| Property | Value |
|----------|-------|
| **Global Minimum** | $f(\mathbf{1}) = 0$ |
| **Search Domain** | $x_i \in [-5, 10]$ |
| **Dimensions** | $n \geq 2$ |
| **Modality** | Unimodal |

```python
from opt.benchmark.functions import rosenbrock
import numpy as np

x = np.array([1, 1])
print(f"f([1, 1]) = {rosenbrock(x)}")  # 0.0
```

---

### Ackley Function

A highly multimodal function with many local minima.

$$
f(\mathbf{x}) = -20\exp\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right) + 20 + e
$$

| Property | Value |
|----------|-------|
| **Global Minimum** | $f(\mathbf{0}) = 0$ |
| **Search Domain** | $x_i \in [-32.768, 32.768]$ |
| **Dimensions** | 2 (standard) |
| **Modality** | Multimodal |

```python
from opt.benchmark.functions import ackley
import numpy as np

x = np.array([0, 0])
print(f"f([0, 0]) = {ackley(x):.10f}")  # ≈ 0
```

---

### Shifted Ackley Function

Non-centered variant of Ackley for more challenging optimization.

```python
from opt.benchmark.functions import shifted_ackley
import numpy as np

x = np.array([1, 0.5])  # Shifted optimum
print(f"f([1, 0.5]) = {shifted_ackley(x):.10f}")  # ≈ 0
```

---

### Rastrigin Function

A highly multimodal function with regularly distributed local minima.

$$
f(\mathbf{x}) = 10n + \sum_{i=1}^{n}\left[x_i^2 - 10\cos(2\pi x_i)\right]
$$

| Property | Value |
|----------|-------|
| **Global Minimum** | $f(\mathbf{0}) = 0$ |
| **Search Domain** | $x_i \in [-5.12, 5.12]$ |
| **Dimensions** | Any |
| **Modality** | Highly multimodal |

```python
from opt.benchmark.functions import rastrigin
import numpy as np

x = np.array([0, 0])
print(f"f([0, 0]) = {rastrigin(x)}")  # 0.0
```

---

### Schwefel Function

Deceptive function with global minimum far from local minima.

$$
f(\mathbf{x}) = 418.9829n - \sum_{i=1}^{n}x_i\sin(\sqrt{|x_i|})
$$

| Property | Value |
|----------|-------|
| **Global Minimum** | $f(420.9687,...) \approx 0$ |
| **Search Domain** | $x_i \in [-500, 500]$ |
| **Dimensions** | Any |
| **Modality** | Multimodal, deceptive |

```python
from opt.benchmark.functions import schwefel
import numpy as np

x = np.array([420.9687, 420.9687])
print(f"f(x) = {schwefel(x):.4f}")
```

---

### Griewank Function

Multimodal with regular structure.

$$
f(\mathbf{x}) = 1 + \frac{1}{4000}\sum_{i=1}^{n}x_i^2 - \prod_{i=1}^{n}\cos\left(\frac{x_i}{\sqrt{i}}\right)
$$

| Property | Value |
|----------|-------|
| **Global Minimum** | $f(\mathbf{0}) = 0$ |
| **Search Domain** | $x_i \in [-600, 600]$ |
| **Dimensions** | Any |
| **Modality** | Multimodal |

```python
from opt.benchmark.functions import griewank
import numpy as np

x = np.array([0, 0])
print(f"f([0, 0]) = {griewank(x)}")  # 0.0
```

---

### Himmelblau Function

Four identical local minima.

$$
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
$$

| Property | Value |
|----------|-------|
| **Global Minima** | 4 points, $f = 0$ |
| **Search Domain** | $x, y \in [-5, 5]$ |
| **Dimensions** | 2 |
| **Modality** | Multimodal |

```python
from opt.benchmark.functions import himmelblau
import numpy as np

# One of four minima
x = np.array([3, 2])
print(f"f([3, 2]) = {himmelblau(x)}")  # 0.0
```

---

## Complete Function List

| Function | Optimal Value | Modality | Separable |
|----------|---------------|----------|-----------|
| `sphere` | 0 | Unimodal | Yes |
| `rosenbrock` | 0 | Unimodal | No |
| `ackley` | 0 | Multimodal | Yes |
| `shifted_ackley` | 0 | Multimodal | Yes |
| `rastrigin` | 0 | Multimodal | Yes |
| `schwefel` | 0 | Multimodal | Yes |
| `griewank` | 0 | Multimodal | No |
| `levi` | 0 | Multimodal | No |
| `himmelblau` | 0 | Multimodal | No |
| `eggholder` | -959.64 | Multimodal | No |
| `beale` | 0 | Multimodal | No |
| `goldstein_price` | 3 | Multimodal | No |
| `booth` | 0 | Unimodal | No |
| `bukin` | 0 | Multimodal | No |
| `matyas` | 0 | Unimodal | No |
| `levi_n13` | 0 | Multimodal | No |
| `three_hump_camel` | 0 | Multimodal | No |
| `easom` | -1 | Multimodal | Yes |
| `cross_in_tray` | -2.06 | Multimodal | No |
| `hold_table` | -19.21 | Multimodal | No |
| `mccormick` | -1.91 | Multimodal | No |

## Usage Example

```python
from opt.benchmark.functions import (
    sphere, rosenbrock, ackley, rastrigin, 
    schwefel, griewank, himmelblau
)
from opt.swarm_intelligence import ParticleSwarm

functions = [
    ("Sphere", sphere, (-5, 5)),
    ("Rosenbrock", rosenbrock, (-5, 10)),
    ("Ackley", ackley, (-5, 5)),
    ("Rastrigin", rastrigin, (-5.12, 5.12)),
]

for name, func, bounds in functions:
    optimizer = ParticleSwarm(
        func=func,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        dim=2,
        max_iter=500,
    )
    _, fitness = optimizer.search()
    print(f"{name}: {fitness:.6e}")
```
