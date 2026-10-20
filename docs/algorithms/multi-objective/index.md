# 📈 Multi-Objective Algorithms

Pareto-optimal solutions for multiple objectives.

This category contains **3 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Multi-Objective Evolutionary Algorithm based on Decomposition](./moead) | MOEA-D | 2007 | Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D). |
| [Non-dominated Sorting Genetic Algorithm II](./nsga-ii) | NSGA-II | 2002 | Non-dominated Sorting Genetic Algorithm II (NSGA-II) multi-objective optimizer. |
| [Strength Pareto Evolutionary Algorithm 2](./spea2) | SPEA2 | 2001 | Strength Pareto Evolutionary Algorithm 2 (SPEA2) multi-objective optimizer. |

## Usage

Every optimizer shares the same interface:

```python
from opt.multi_objective.moead import MOEAD
from opt.benchmark.functions import sphere

optimizer = MOEAD(
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
