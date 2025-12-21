# Useful Optimizer

[![DOI](https://zenodo.org/badge/776526436.svg)](https://zenodo.org/doi/10.5281/zenodo.13294276)
[![PyPI version](https://badge.fury.io/py/useful-optimizer.svg)](https://badge.fury.io/py/useful-optimizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive collection of optimization algorithms for numeric problems.**

Useful Optimizer provides 100+ implementations of state-of-the-art optimization algorithms organized into logical categories, making it easy to discover, compare, and integrate the right optimizer for your problem.

---

## :rocket: Quick Start

=== "pip"

    ```bash
    pip install git+https://github.com/Anselmoo/useful-optimizer
    ```

=== "uv (recommended)"

    ```bash
    uv add git+https://github.com/Anselmoo/useful-optimizer
    ```

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

optimizer = ParticleSwarm(
    func=shifted_ackley,
    dim=2,
    lower_bound=-12.768,
    upper_bound=+12.768,
    population_size=100,
    max_iter=1000,
)
best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

---

## :zap: Features

<div class="grid cards" markdown>

-   :material-bee:{ .lg .middle } **Swarm Intelligence**

    ---

    57 nature-inspired algorithms including Particle Swarm Optimization, Ant Colony, Grey Wolf Optimizer, and many more.

    [:octicons-arrow-right-24: Explore](algorithms/swarm-intelligence.md)

-   :material-dna:{ .lg .middle } **Evolutionary Algorithms**

    ---

    Genetic algorithms, CMA-ES, Differential Evolution, and other population-based methods.

    [:octicons-arrow-right-24: Explore](algorithms/evolutionary.md)

-   :material-function-variant:{ .lg .middle } **Gradient-Based**

    ---

    Adam, AdaGrad, RMSprop, SGD with momentum, and other gradient descent variants.

    [:octicons-arrow-right-24: Explore](algorithms/gradient-based.md)

-   :material-target:{ .lg .middle } **Classical Methods**

    ---

    BFGS, L-BFGS, Nelder-Mead, Simulated Annealing, and other well-established techniques.

    [:octicons-arrow-right-24: Explore](algorithms/classical.md)

-   :material-atom:{ .lg .middle } **Physics-Inspired**

    ---

    Gravitational Search, Equilibrium Optimizer, and other physics-based algorithms.

    [:octicons-arrow-right-24: Explore](algorithms/physics-inspired.md)

-   :material-chart-bell-curve:{ .lg .middle } **Probabilistic**

    ---

    Bayesian Optimization, Parzen Tree Estimators, and probabilistic approaches.

    [:octicons-arrow-right-24: Explore](algorithms/probabilistic.md)

</div>

---

## :books: Algorithm Categories

| Category | Count | Description |
|----------|-------|-------------|
| [Swarm Intelligence](algorithms/swarm-intelligence.md) | 57 | Nature-inspired collective behavior algorithms |
| [Evolutionary](algorithms/evolutionary.md) | 6 | Population-based evolutionary algorithms |
| [Gradient-Based](algorithms/gradient-based.md) | 11 | Gradient descent and variants |
| [Classical](algorithms/classical.md) | 9 | Traditional optimization methods |
| [Metaheuristic](algorithms/metaheuristic.md) | 14 | High-level problem-independent frameworks |
| [Physics-Inspired](algorithms/physics-inspired.md) | 4 | Physics-based optimization |
| [Social-Inspired](algorithms/social-inspired.md) | 4 | Social behavior-based algorithms |
| [Probabilistic](algorithms/probabilistic.md) | 5 | Probabilistic and Bayesian methods |
| [Constrained](algorithms/constrained.md) | 5 | Constrained optimization methods |
| [Multi-Objective](algorithms/multi-objective.md) | 4 | Multi-objective optimization |

---

## :test_tube: Benchmark Functions

Useful Optimizer includes a comprehensive suite of benchmark functions for testing and comparing optimizer performance:

- **Sphere** - Simple convex function
- **Rosenbrock** - Unimodal, non-convex valley
- **Ackley** - Multimodal with many local minima
- **Rastrigin** - Highly multimodal
- **Schwefel** - Deceptive global minimum
- **And many more...**

[:octicons-arrow-right-24: View all benchmark functions](benchmarks/functions.md)

---

## :chart_with_upwards_trend: Scientific Visualization

The documentation includes research-grade visualizations following academic benchmarking standards (COCO, IOHprofiler):

- **ECDF Curves** - Empirical Cumulative Distribution Functions
- **Convergence Plots** - With confidence bands
- **Statistical Comparisons** - Friedman test heatmaps, Wilcoxon matrices
- **Interactive Dashboards** - 3D fitness landscapes, search trajectories

[:octicons-arrow-right-24: View benchmarks](benchmarks/index.md)

---

## :handshake: Contributing

Contributions to Useful Optimizer are welcome! Please read the [contributing guidelines](https://github.com/Anselmoo/useful-optimizer/blob/main/CONTRIBUTING.md) before getting started.

---

## :page_facing_up: License

Useful Optimizer is released under the [MIT License](https://opensource.org/licenses/MIT).

---

!!! warning "Disclaimer"

    This project was generated with GitHub Copilot and may not be completely verified. Please use with caution and feel free to report any issues you encounter.

!!! note "Legacy Random API"

    Some parts still contain the legacy `np.random.rand` call. See [NumPy documentation](https://docs.astral.sh/ruff/rules/numpy-legacy-random/) for more details.
