# 🎯 Classical Algorithms

Traditional mathematical optimization methods.

This category contains **9 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Broyden-Fletcher-Goldfarb-Shanno](./bfgs) | BFGS | 1970 | Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton optimization algorithm. |
| [Conjugate Gradient Method](./conjugate-gradient) | CG | 1952 | Conjugate Gradient Method (CG) optimization algorithm. |
| [Hill Climbing](./hill-climbing) | HC | 1958 | Hill Climbing local search optimization algorithm. |
| [Limited-memory BFGS](./lbfgs) | L-BFGS | 1980 | Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm. |
| [Nelder-Mead Simplex Method](./nelder-mead) | NM | 1965 | Nelder-Mead Simplex Method optimization algorithm. |
| [Powell's Conjugate Direction Method](./powell) | POWELL | 1964 | Powell's Conjugate Direction Method optimization algorithm. |
| [Simulated Annealing](./simulated-annealing) | SA | 1983 | Simulated Annealing (SA) metaheuristic optimization algorithm. |
| [Tabu Search](./tabu-search) | TS | 1986 | Tabu Search metaheuristic optimization algorithm. |
| [Trust Region Method](./trust-region) | TR | 1983 | Trust Region optimization algorithm. |

## Usage

Every optimizer shares the same interface:

```python
from opt.classical.bfgs import BFGS
from opt.benchmark.functions import sphere

optimizer = BFGS(
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
