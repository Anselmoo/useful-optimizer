# Algorithms Overview

<!-- BEGIN GENERATED: catalog (scripts/generate_docs.py) -->

Useful Optimizer provides **117 optimization algorithms** in 10 categories. Each one shares the same interface and is documented with its mathematical formulation, parameters, references and benchmark results.

## Algorithm Categories

| Category | Algorithms | Description |
|----------|-----------|-------------|
| [🦋 Swarm Intelligence](./swarm-intelligence/) | 56 | Nature-inspired algorithms based on collective behavior |
| [🧬 Evolutionary](./evolutionary/) | 6 | Algorithms inspired by biological evolution |
| [🧠 Gradient-Based](./gradient-based/) | 11 | Optimization using gradient information |
| [🎯 Classical](./classical/) | 9 | Traditional mathematical optimization methods |
| [🔧 Metaheuristic](./metaheuristic/) | 14 | High-level problem-independent algorithms |
| [⚛️ Physics-Inspired](./physics-inspired/) | 4 | Algorithms based on physical phenomena |
| [📊 Probabilistic](./probabilistic/) | 5 | Stochastic and Bayesian optimization methods |
| [👥 Social-Inspired](./social-inspired/) | 4 | Algorithms inspired by social behaviors |
| [🔒 Constrained](./constrained/) | 5 | Optimization with constraints |
| [📈 Multi-Objective](./multi-objective/) | 3 | Pareto-optimal solutions for multiple objectives |

<!-- END GENERATED: catalog -->

## Choosing an Algorithm

### By Problem Type

| Problem Type | Recommended Algorithms |
|--------------|----------------------|
| **Smooth, unimodal** | BFGS, L-BFGS, Conjugate Gradient |
| **Multi-modal** | PSO, DE, CMA-ES, Firefly |
| **High-dimensional** | CMA-ES, DE, Grey Wolf |
| **Noisy objective** | PSO, DE, SA |
| **Constrained** | Augmented Lagrangian, SLP |
| **Black-box** | Nelder-Mead, PSO, DE |

### By Computational Budget

| Budget | Recommended Algorithms |
|--------|----------------------|
| **Very limited** | Nelder-Mead, Hill Climbing |
| **Medium** | PSO, DE, Grey Wolf |
| **Large** | CMA-ES, multi-start BFGS |

## Common Interface

All algorithms share the same interface:

```python
from opt.swarm_intelligence import ParticleSwarm

optimizer = ParticleSwarm(
    func=objective_function,    # Callable[[np.ndarray], float]
    lower_bound=-10.0,          # Lower bound of search space
    upper_bound=10.0,           # Upper bound of search space
    dim=10,                     # Number of dimensions
    max_iter=100                # Maximum iterations
)

best_solution, best_fitness = optimizer.search()
```

## Performance Comparison

See the [Benchmarks](/benchmarks/) section for detailed performance comparisons on standard test functions.
