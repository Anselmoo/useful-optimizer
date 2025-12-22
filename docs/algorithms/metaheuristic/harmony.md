# Harmony Search

<span class="badge badge-metaheuristic">Metaheuristic</span>

Harmony Search (HS) algorithm.

## Algorithm Overview

This module implements the Harmony Search optimization algorithm. Harmony Search is a
metaheuristic algorithm inspired by the improvisation process of musicians. It is
commonly used for solving optimization problems.

The HarmonySearch class is the main class that implements the algorithm. It takes an
objective function, lower and upper bounds of the search space, dimensionality of the
search space, and other optional parameters. The search method runs the optimization
process and returns the best solution found and its fitness value.

## Usage

```python
from opt.metaheuristic.harmony_search import HarmonySearch
from opt.benchmark.functions import sphere

optimizer = HarmonySearch(
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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `population_size` | `int` | `100` | The size of the population. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `harmony_memory_accepting_rate` | `float` | `0.95` | The rate at which the harmony memory is accepted. |
| `pitch_adjusting_rate` | `float` | `0.7` | The rate at which the pitch is adjusted. |
| `bandwidth` | `float` | `0.01` | The bandwidth for adjusting the pitch. |
| `seed` | `int  \|  None` | `None` | The seed for the random number generator. |

## See Also

- [Metaheuristic Algorithms](/algorithms/metaheuristic/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`harmony_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/metaheuristic/harmony_search.py)
:::
