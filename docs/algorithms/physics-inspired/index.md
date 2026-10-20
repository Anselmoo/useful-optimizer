# ⚛️ Physics-Inspired Algorithms

Algorithms based on physical phenomena.

This category contains **4 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Atom Search Optimization](./atom-search) | ASO | 2019 | Atom Search Optimization (ASO) algorithm. |
| [Equilibrium Optimizer](./equilibrium-optimizer) | EO | 2020 | Equilibrium Optimizer (EO) optimization algorithm. |
| [Gravitational Search Algorithm](./gravitational-search) | GSA | 2009 | Gravitational Search Algorithm (GSA) optimization algorithm. |
| [RIME: A Physics-based Optimization](./rime-optimizer) | RIME | 2023 | RIME Optimization Algorithm (RIME) optimization algorithm. |

## Usage

Every optimizer shares the same interface:

```python
from opt.physics_inspired.atom_search import AtomSearchOptimizer
from opt.benchmark.functions import sphere

optimizer = AtomSearchOptimizer(
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
