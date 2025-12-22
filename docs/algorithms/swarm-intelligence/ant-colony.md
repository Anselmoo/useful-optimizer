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
| `func` | `Callable` | Required | The objective function to be minimized. |
| `lower_bound` | `float` | Required | The lower bound of the search space. |
| `upper_bound` | `float` | Required | The upper bound of the search space. |
| `dim` | `int` | Required | The dimensionality of the search space. |
| `max_iter` | `int` | `1000` | The maximum number of iterations. |
| `population_size` | `int` | `100` | The number of ants in the colony. |
| `seed` | `int  \|  None` | `None` | The seed value for random number generation. |
| `alpha` | `float` | `1` | The importance of pheromone in the solution construction. |
| `beta` | `float` | `1` | The importance of heuristic information in the solution construction. |
| `rho` | `float` | `0.5` | The pheromone evaporation rate. |
| `q` | `float` | `1` | The pheromone deposit factor. |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`ant_colony.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/ant_colony.py)
:::
