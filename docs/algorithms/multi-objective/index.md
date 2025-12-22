# Multi-Objective Optimization Algorithms

Algorithms designed to optimize multiple conflicting objectives simultaneously.

## Overview

Multi-objective optimization algorithms find a set of Pareto-optimal solutions rather than a single optimum. These solutions represent trade-offs between competing objectives.

## Available Algorithms

- [NSGA-II](./nsga-ii) - Non-dominated Sorting Genetic Algorithm II
- [SPEA2](./spea2) - Strength Pareto Evolutionary Algorithm 2
- [MOEA/D](./moead) - Multi-Objective Evolutionary Algorithm based on Decomposition

## Key Concepts

**Pareto Optimality**: A solution is Pareto-optimal if no other solution is better in all objectives.

**Pareto Front**: The set of all Pareto-optimal solutions in objective space.

**Dominance**: Solution A dominates solution B if A is at least as good as B in all objectives and strictly better in at least one.

## Usage Example

```python
from opt.multi_objective import NSGAII

def objective1(x):
    return sum(x**2)  # Minimize

def objective2(x):
    return sum((x-1)**2)  # Minimize

optimizer = NSGAII(
    objectives=[objective1, objective2],
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    max_iter=100,
    population_size=100
)
pareto_front, pareto_solutions = optimizer.search()
```

## Output

Multi-objective algorithms return:
- **Pareto Front**: Array of objective values for non-dominated solutions
- **Pareto Solutions**: Array of decision variables for non-dominated solutions

## See Also

- [API Reference](/api/) - Complete API documentation
- [Evolutionary Algorithms](../evolutionary/) - Single-objective evolutionary methods
