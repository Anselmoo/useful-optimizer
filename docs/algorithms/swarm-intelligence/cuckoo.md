# Cuckoo Search

<span class="badge badge-swarm">Swarm Intelligence</span>

Cuckoo Search Optimization Algorithm.

## Algorithm Overview

This module implements the Cuckoo Search (CS) optimization algorithm.
CS is a nature-inspired metaheuristic algorithm, which is based on the obligate brood
parasitism of some cuckoo species. In these species, the cuckoos lay their eggs in the
nests of other host birds. If the host bird discovers the eggs are not their own, it
will either throw these alien eggs away or abandon its nest and build a completely new
one.

In the context of the CS algorithm, each egg in a nest represents a solution, and a
cuckoo egg represents a new solution. The aim is to use the new and potentially better
solutions (cuckoo eggs) to replace a not-so-good solution in the nests. In the simplest
form, each nest represents a solution, and thus the egg represents a new solution that
is to replace the old one if the new solution is better.

The CS algorithm is used to solve optimization problems by iteratively trying to
improve a candidate solution with regard to a given measure of quality, or fitness
function.

## Usage

```python
from opt.swarm_intelligence.cuckoo_search import CuckooSearch
from opt.benchmark.functions import sphere

optimizer = CuckooSearch(
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
| `population_size` | `int` | `100` | Number of individuals in population |
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `mutation_probability` | `float` | `0.1` | Algorithm-specific parameter |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |

## See Also

- [Swarm Intelligence Algorithms](/algorithms/swarm-intelligence/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`cuckoo_search.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/swarm_intelligence/cuckoo_search.py)
:::
