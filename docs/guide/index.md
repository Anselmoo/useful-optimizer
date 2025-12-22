# Introduction

Useful Optimizer is a comprehensive Python library containing **54+ optimization algorithms** for solving numeric problems. The library is designed to be easy to use while providing research-grade performance and flexibility.

## Why Useful Optimizer?

- **Comprehensive Coverage**: From swarm intelligence to gradient-based methods, covering all major optimization paradigms
- **Consistent API**: All optimizers follow the same interface for easy experimentation
- **Well Documented**: Every algorithm includes docstrings following Google style conventions
- **Research Ready**: Includes benchmark functions and visualization tools for academic use
- **Pure Python**: Easy to understand, modify, and extend

## Algorithm Categories

| Category | Count | Description |
|----------|-------|-------------|
| Swarm Intelligence | 57+ | Nature-inspired population-based algorithms |
| Evolutionary | 6 | Evolution-based optimization methods |
| Gradient-Based | 11 | Gradient descent variants and adaptive methods |
| Classical | 9 | Traditional mathematical optimization |
| Metaheuristic | 12 | Problem-independent optimization frameworks |
| Constrained | 2 | Methods for constrained optimization |
| Probabilistic | 2 | Probability-based optimization |

## Key Features

### Unified Interface

All optimizers implement the `AbstractOptimizer` base class with a consistent `search()` method:

```python
from opt.swarm_intelligence import ParticleSwarm

optimizer = ParticleSwarm(
    func=objective_function,
    lower_bound=-10.0,
    upper_bound=10.0,
    dim=10,
    max_iter=100
)

best_solution, best_fitness = optimizer.search()
```

### Multiple Import Styles

```python
# Categorical imports (recommended)
from opt.swarm_intelligence import ParticleSwarm
from opt.gradient_based import AdamW

# Direct imports (backward compatible)
from opt import ParticleSwarm, AdamW
```

### Built-in Benchmark Functions

```python
from opt.benchmark.functions import (
    sphere,
    rosenbrock,
    rastrigin,
    ackley,
    shifted_ackley,
    griewank
)
```

## Getting Started

1. **[Installation](./installation)** - Set up the library
2. **[Quick Start](./quickstart)** - Run your first optimization
3. **[Advanced Usage](./advanced)** - Customize and extend

## Community

- [GitHub Repository](https://github.com/Anselmoo/useful-optimizer)
- [Issue Tracker](https://github.com/Anselmoo/useful-optimizer/issues)
- [Zenodo DOI](https://zenodo.org/doi/10.5281/zenodo.13294276)
