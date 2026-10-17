# 🧬 Evolutionary Algorithms

Algorithms inspired by biological evolution.

This category contains **6 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Covariance Matrix Adaptation Evolution Strategy](./cma-es) | CMA-ES | 2001 | Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm. |
| [Cultural Algorithm](./cultural-algorithm) | CA | 1994 | Cultural Algorithm (CA) optimization algorithm. |
| [Differential Evolution](./differential-evolution) | DE | 1997 | Differential Evolution (DE) optimization algorithm. |
| [Estimation of Distribution Algorithm](./estimation-of-distribution-algorithm) | EDA | 1996 | Estimation of Distribution Algorithm (EDA) optimization algorithm. |
| [Genetic Algorithm](./genetic-algorithm) | GA | 1975 | Genetic Algorithm (GA) optimization algorithm. |
| [Imperialist Competitive Algorithm](./imperialist-competitive-algorithm) | ICA | 2007 | Imperialist Competitive Algorithm (ICA) optimization algorithm. |

## Usage

Every optimizer shares the same interface:

```python
from opt.evolutionary.cma_es import CMAESAlgorithm
from opt.benchmark.functions import sphere

optimizer = CMAESAlgorithm(
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
