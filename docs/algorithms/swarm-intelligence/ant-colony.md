# Ant Colony Optimization

<span class="badge badge-swarm">Swarm Intelligence</span>

Ant Colony Optimization (ACO) Algorithm.

## Algorithm Overview

This module implements the Ant Colony Optimization (ACO) algorithm. ACO is a
population-based metaheuristic that can be used to find approximate solutions to
difficult optimization problems.

In ACO, a set of software agents called artificial ants search for good solutions to a
given optimization problem. To apply ACO, the optimization problem is transformed into
the problem of finding the best path on a weighted graph. The artificial ants
incrementally build solutions by moving on the graph. The solution construction process
 is stochastic and is biased by a pheromone model, that is, a set of parameters
associated with graph components (either nodes or edges) whose values are modified
at runtime by the ants.

ACO is particularly useful for problems that can be reduced to finding paths on
weighted graphs, like the traveling salesman problem, the vehicle routing problem, and
the quadratic assignment problem.

## Usage

```python
from opt.swarm_intelligence.ant_colony import AntColony
from opt.benchmark.functions import sphere

optimizer = AntColony(
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
| `population_size` | `int` | `100` | Number of ants in colony. |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility. |
| `alpha` | `float` | `1` | Pheromone influence exponent. |
| `beta` | `float` | `1` | Heuristic information weight (not used in basic
        continuous ACO). |
| `rho` | `float` | `0.5` | Pheromone evaporation rate in [0, 1]. |
| `q` | `float` | `1` | Pheromone deposit constant. |
| `track_history` | `bool` | `False` | Track optimization history for visualization |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`ant_colony.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/ant_colony.py)
:::
