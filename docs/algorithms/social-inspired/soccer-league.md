# Soccer League Optimizer

<span class="badge badge-social">Social-Inspired</span>

Soccer League Competition Algorithm.

## Algorithm Overview

This module implements the Soccer League Competition (SLC) algorithm,
a social-inspired metaheuristic based on soccer league dynamics.

The algorithm simulates soccer team behaviors including matches,
transfers, and training processes.

## Reference

> Moosavian, N., & Roodsari, B. K. (2014). Soccer League Competition Algorithm: A novel meta-heuristic algorithm for optimal design of water distribution networks. Swarm and Evolutionary Computation, 17, 14-24. DOI: 10.1016/j.swevo.2014.02.002

[📄 View Paper (DOI: 10.1016/j.swevo.2014.02.002)](https://doi.org/10.1016/j.swevo.2014.02.002)

## Usage

```python
from opt.social_inspired.soccer_league_optimizer import SoccerLeagueOptimizer
from opt.benchmark.functions import sphere

optimizer = SoccerLeagueOptimizer(
    func=sphere,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=500,
    population_size=50,
)

best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.6e}")
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | `Callable` | Required | Objective function to minimize |
| `lower_bound` | `float` | Required | Lower bound of search space |
| `upper_bound` | `float` | Required | Upper bound of search space |
| `dim` | `int` | Required | Problem dimensionality |
| `population_size` | `int` | `30` | Number of individuals in population |
| `max_iter` | `int` | `100` | Maximum number of iterations |
| `num_teams` | `int` | `10` | Algorithm-specific parameter |

## See Also

- [Social-Inspired Algorithms](/algorithms/social-inspired/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`soccer_league_optimizer.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/social_inspired/soccer_league_optimizer.py)
:::
