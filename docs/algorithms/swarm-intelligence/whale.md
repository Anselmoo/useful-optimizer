# Whale Optimization Algorithm

<span class="badge badge-swarm">Swarm Intelligence</span>

Whale Optimization Algorithm (WOA).

## Algorithm Overview

This module implements the Whale Optimization Algorithm (WOA). WOA is a metaheuristic
optimization algorithm inspired by the hunting behavior of humpback whales.
The algorithm is based on the echolocation behavior of humpback whales, which use sounds
to communicate, navigate and hunt in dark or murky waters.

In WOA, each whale represents a potential solution, and the objective function
determines the quality of the solutions. The whales try to update their positions by
mimicking the hunting behavior of humpback whales, which includes encircling,
bubble-net attacking, and searching for prey.

WOA has been used for various kinds of optimization problems including function
optimization, neural network training, and other areas of engineering.

## Usage

```python
from opt.swarm_intelligence.whale_optimization_algorithm import WhaleOptimizationAlgorithm
from opt.benchmark.functions import sphere

optimizer = WhaleOptimizationAlgorithm(
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
| `func` | `Callable` | Required | Objective function to minimize. |
| `lower_bound` | `float` | Required | Lower bound of search space. |
| `upper_bound` | `float` | Required | Upper bound of search space. |
| `dim` | `int` | Required | Problem dimensionality. |
| `max_iter` | `int` | `1000` | Maximum iterations. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `population_size` | `int` | `100` | Number of whales. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`whale_optimization_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/whale_optimization_algorithm.py)
:::
