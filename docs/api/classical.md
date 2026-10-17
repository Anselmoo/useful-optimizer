# Classical API

API reference for the 9 classes of `opt.classical`. This page is generated from the source; see the [algorithm pages](/algorithms/classical/) for formulations and references.

## Imports

```python
from opt.classical import (
    BFGS,
    ConjugateGradient,
    HillClimbing,
    LBFGS,
    NelderMead,
    Powell,
    SimulatedAnnealing,
    TabuSearch,
    TrustRegion,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`BFGS`](/algorithms/classical/bfgs) | Broyden-Fletcher-Goldfarb-Shanno | `num_restarts=25` |
| [`ConjugateGradient`](/algorithms/classical/conjugate-gradient) | Conjugate Gradient Method | `num_restarts=10` |
| [`HillClimbing`](/algorithms/classical/hill-climbing) | Hill Climbing | `initial_step_sizes=1.0`, `acceleration=1.2`, `epsilon=1e-06` |
| [`LBFGS`](/algorithms/classical/lbfgs) | Limited-memory BFGS | `num_restarts=25` |
| [`NelderMead`](/algorithms/classical/nelder-mead) | Nelder-Mead Simplex Method | `num_restarts=25` |
| [`Powell`](/algorithms/classical/powell) | Powell's Conjugate Direction Method | `num_restarts=10` |
| [`SimulatedAnnealing`](/algorithms/classical/simulated-annealing) | Simulated Annealing | `init_temperature=1000`, `stopping_temperature=1e-08`, `cooling_rate=0.99` |
| [`TabuSearch`](/algorithms/classical/tabu-search) | Tabu Search | `tabu_list_size=50`, `neighborhood_size=10` |
| [`TrustRegion`](/algorithms/classical/trust-region) | Trust Region Method | `num_restarts=10`, `method='trust-constr'` |
