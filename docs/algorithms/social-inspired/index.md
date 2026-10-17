# 👥 Social-Inspired Algorithms

Algorithms inspired by social behaviors.

This category contains **4 algorithms**.

| Algorithm | Acronym | Year | Description |
|-----------|---------|------|-------------|
| [Political Optimizer](./political-optimizer) | PO | 2020 | Political Optimizer (PO) algorithm. |
| [Soccer League Competition](./soccer-league-optimizer) | SLC | 2014 | Soccer League Competition (SLC) algorithm. |
| [Social Group Optimization](./social-group-optimizer) | SGO | 2016 | Social Group Optimization (SGO) algorithm. |
| [Teaching-Learning Based Optimization](./teaching-learning) | TLBO | 2011 | Teaching-Learning Based Optimization (TLBO) algorithm. |

## Usage

Every optimizer shares the same interface:

```python
from opt.social_inspired.political_optimizer import PoliticalOptimizer
from opt.benchmark.functions import sphere

optimizer = PoliticalOptimizer(
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
