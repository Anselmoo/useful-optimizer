# Probabilistic Optimization API

API reference for probabilistic optimization algorithms in `opt.probabilistic`.

## Module Overview

```python
from opt.probabilistic import (
    EstimationOfDistribution,
    CrossEntropy,
    BayesianOptimization,
)
```

## Common Interface

```python
class ProbabilisticOptimizer(AbstractOptimizer):
    def __init__(
        self,
        func: Callable,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 50,
        **kwargs
    ):
        pass

    def search(self) -> tuple[np.ndarray, float]:
        pass
```

## Available Algorithms

- `EstimationOfDistribution` - Builds probability distribution of solutions
- `CrossEntropy` - Adaptive importance sampling
- `BayesianOptimization` - Gaussian process-based

## Example Usage

```python
from opt.probabilistic import EstimationOfDistribution
from opt.benchmark.functions import rastrigin

# Estimation of Distribution Algorithm
eda = EstimationOfDistribution(
    func=rastrigin,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=100,
    population_size=100,
    selection_size=20  # Number of best individuals for distribution update
)
solution, fitness = eda.search()
print(f"Best fitness: {fitness:.6e}")
```

## Bayesian Optimization Example

```python
from opt.probabilistic import BayesianOptimization
from opt.benchmark.functions import ackley

# Bayesian Optimization (typically for expensive functions)
bo = BayesianOptimization(
    func=ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=5,  # Lower dimensions recommended for BO
    max_iter=50,  # Fewer iterations than other methods
    n_initial_points=10,
    acquisition='ei'  # Expected Improvement
)
solution, fitness = bo.search()
```

## See Also

- [Probabilistic Optimization](/algorithms/probabilistic/) - Algorithm details
