# Genetic Algorithm

<span class="badge badge-evolutionary">Evolutionary</span>

Genetic Algorithm Optimizer.

## Algorithm Overview

This module implements a genetic algorithm (GA) optimizer. Genetic algorithms are a
part of evolutionary computing, which is a rapidly growing area of artificial
intelligence.

The GA optimizer starts with a population of candidate solutions to an optimization
problem and evolves this population by iteratively applying a set of genetic operators.

Key components of the GA optimizer include:
- Initialization: The population is initialized with a set of random solutions.
- Selection: Solutions are selected to reproduce based on their fitness. The better the
    solutions, the more chances they have to reproduce.
- Crossover (or recombination): Pairs of solutions are selected for reproduction to
    create one or more offspring, in which each offspring consists of a mix of the
    parents' traits.
- Mutation: After crossover, the offspring are mutated with a small probability.
    Mutation introduces small changes in the solutions, providing genetic diversity.
- Replacement: The population is updated to include the new, fitter solutions.

The GA optimizer is suitable for solving both constrained and unconstrained optimization
problems. It's particularly useful for problems where the search space is large and
complex, and where traditional optimization methods may not be applicable.

## Usage

```python
from opt.evolutionary.genetic_algorithm import GeneticAlgorithm
from opt.benchmark.functions import sphere

optimizer = GeneticAlgorithm(
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
| `population_size` | `int` | `150` | Number of individuals in population |
| `max_iter` | `int` | `1000` | Maximum number of iterations |
| `tournament_size` | `int` | `3` | Algorithm-specific parameter |
| `crossover_rate` | `float` | `0.7` | Probability of crossover |
| `seed` | `int  \|  None` | `None` | Random seed for reproducibility |

## See Also

- [Evolutionary Algorithms](/algorithms/evolutionary/)
- [All Algorithms](/algorithms/)
- [Benchmark Functions](/api/benchmark-functions)

---

::: tip Source Code
View the implementation: [`genetic_algorithm.py`](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/evolutionary/genetic_algorithm.py)
:::
