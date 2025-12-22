# Classical Algorithms API

API reference for classical optimization algorithms in `opt.classical`.

## Module Overview

```python
from opt.classical import (
    BFGS,
    NelderMead,
    SimulatedAnnealing,
    HillClimbing,
    Powell,
    TrustRegion,
)
```

## Common Interface

```python
class ClassicalAlgorithm(AbstractOptimizer):
    def __init__(
        self,
        func: Callable,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        **kwargs
    ):
        pass
    
    def search(self) -> tuple[np.ndarray, float]:
        pass
```

## Available Algorithms

- `BFGS` - Quasi-Newton method
- `NelderMead` - Simplex-based derivative-free
- `SimulatedAnnealing` - Probabilistic metaheuristic
- `HillClimbing` - Local search
- `Powell` - Conjugate direction method
- `TrustRegion` - Constrained optimization

## Example Usage

```python
from opt.classical import NelderMead, SimulatedAnnealing
from opt.benchmark.functions import rosenbrock

# Nelder-Mead
nm = NelderMead(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=10,
    dim=10,
    max_iter=1000
)
solution, fitness = nm.search()

# Simulated Annealing
sa = SimulatedAnnealing(
    func=rosenbrock,
    lower_bound=-5,
    upper_bound=10,
    dim=10,
    max_iter=1000,
    initial_temp=100,
    cooling_rate=0.95
)
solution, fitness = sa.search()
```

## See Also

- [Classical Algorithms](/algorithms/classical/) - Algorithm details
