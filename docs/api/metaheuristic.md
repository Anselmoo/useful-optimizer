# Metaheuristic Algorithms API

API reference for metaheuristic algorithms in `opt.metaheuristic`.

## Module Overview

```python
from opt.metaheuristic import (
    HarmonySearch,
    CrossEntropy,
    SineCosine,
    TabuSearch,
    VariableNeighborhood,
)
```

## Common Interface

```python
class MetaheuristicAlgorithm(AbstractOptimizer):
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

- `HarmonySearch` - Music-inspired optimization
- `CrossEntropy` - Adaptive importance sampling
- `SineCosine` - Mathematical function-based
- `TabuSearch` - Memory-based search
- `VariableNeighborhood` - Local search strategy

## Example Usage

```python
from opt.metaheuristic import HarmonySearch, SineCosine
from opt.benchmark.functions import ackley

# Harmony Search
hs = HarmonySearch(
    func=ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=10,
    max_iter=100,
    harmony_memory_size=30,
    hmcr=0.9,  # Harmony memory consideration rate
    par=0.3    # Pitch adjustment rate
)
solution, fitness = hs.search()

# Sine Cosine Algorithm
sca = SineCosine(
    func=ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=10,
    max_iter=100,
    population_size=30
)
solution, fitness = sca.search()
```

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/) - Algorithm details
