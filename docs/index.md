---
title: Useful Optimizer
description: 58 optimization algorithms for numeric problems - A comprehensive Python library
---

# Useful Optimizer

<p class="subtitle" style="font-size: 1.25rem; color: var(--oklch-fg-73); margin-top: -0.5rem;">
A comprehensive collection of <strong>58 optimization algorithms</strong> for numeric problems
</p>

[![DOI](https://zenodo.org/badge/776526436.svg)](https://zenodo.org/doi/10.5281/zenodo.13294276)
[![PyPI version](https://badge.fury.io/py/useful-optimizer.svg)](https://pypi.org/project/useful-optimizer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## :rocket: Quick Start

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-12.768,
    upper_bound=+12.768,
    dim=2,
    population_size=30,
    max_iter=100,
)
best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

---

## :sparkles: Features

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **58+ Algorithms**

    ---

    Comprehensive collection spanning swarm intelligence, evolutionary, gradient-based, classical, and metaheuristic methods.

    [:octicons-arrow-right-24: Browse Algorithms](algorithms/index.md)

-   :material-puzzle:{ .lg .middle } **Easy Integration**

    ---

    Simple, consistent API across all optimizers. Drop-in replacement for your optimization needs.

    [:octicons-arrow-right-24: Getting Started](getting-started/quickstart.md)

-   :material-chart-line:{ .lg .middle } **Benchmark Suite**

    ---

    Built-in benchmark functions (Ackley, Rosenbrock, Sphere, etc.) for algorithm comparison.

    [:octicons-arrow-right-24: Benchmarks](benchmarks/index.md)

-   :material-book-open-variant:{ .lg .middle } **Well Documented**

    ---

    Mathematical formulations, pseudocode, and usage examples for every algorithm.

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

---

## :bookmark_tabs: Algorithm Categories

| Category | Count | Description |
|----------|-------|-------------|
| [**Swarm Intelligence**](algorithms/swarm-intelligence/index.md) | 12 | PSO, Ant Colony, Firefly, Grey Wolf, etc. |
| [**Gradient-Based**](algorithms/gradient-based/index.md) | 11 | SGD, Adam, AdamW, RMSprop, etc. |
| [**Evolutionary**](algorithms/evolutionary/index.md) | 6 | Genetic Algorithm, CMA-ES, Differential Evolution |
| [**Classical**](algorithms/classical/index.md) | 9 | BFGS, Nelder-Mead, Simulated Annealing |
| [**Metaheuristic**](algorithms/metaheuristic/index.md) | 12 | Harmony Search, Cross Entropy, Eagle Strategy |
| **Others** | 4 | Constrained & Probabilistic methods |

---

## :package: Installation

=== "pip"

    ```bash
    pip install useful-optimizer
    ```

=== "uv (recommended)"

    ```bash
    uv add useful-optimizer
    ```

=== "From source"

    ```bash
    pip install git+https://github.com/Anselmoo/useful-optimizer
    ```

---

## :books: Citation

If you use Useful Optimizer in your research, please cite:

```bibtex
@software{useful_optimizer,
  author = {Hahn, Anselm},
  title = {Useful Optimizer: A Python Library for Optimization Algorithms},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.13294276},
  url = {https://github.com/Anselmoo/useful-optimizer}
}
```

---

## :handshake: Contributing

We welcome contributions! Please see our [contributing guidelines](contributing/guidelines.md) for more information.

---

<div style="text-align: center; margin-top: 2rem;">
  <a href="getting-started/quickstart/" class="md-button md-button--primary">Get Started</a>
  <a href="algorithms/" class="md-button">Browse Algorithms</a>
</div>
