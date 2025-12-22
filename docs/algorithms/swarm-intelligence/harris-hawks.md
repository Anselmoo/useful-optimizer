# Harris Hawks Optimizer

<span class="badge badge-swarm">Swarm Intelligence</span>

Harris Hawks Optimization (HHO) Algorithm.

## Algorithm Overview

This module implements the Harris Hawks Optimization algorithm, a population-based
metaheuristic inspired by the cooperative hunting behavior of Harris hawks in nature.

The algorithm simulates the surprise pounce (or seven kills) strategy where
hawks cooperate to catch prey. It includes exploration and exploitation phases
with different attacking strategies based on the escaping energy of prey.

## Reference

> Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., & Chen, H. (2019). Harris hawks optimization: Algorithm and applications. Future Generation Computer Systems, 97, 849-872. DOI: 10.1016/j.future.2019.02.028

[📄 View Paper (DOI: 10.1016/j.future.2019.02.028)](https://doi.org/10.1016/j.future.2019.02.028)

## Usage

```python
from opt.swarm_intelligence.harris_hawks_optimization import HarrisHawksOptimizer
from opt.benchmark.functions import sphere

optimizer = HarrisHawksOptimizer(
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
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |
| `population_size` | `int` | `100` | Number of individuals in population |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`harris_hawks_optimization.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/harris_hawks_optimization.py)
:::
