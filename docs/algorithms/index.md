# Algorithms Overview

Useful Optimizer provides **54+ optimization algorithms** organized into logical categories. Each algorithm is designed to solve numeric optimization problems with different characteristics.

## Algorithm Categories

### 🦋 Swarm Intelligence (57+ algorithms)

Nature-inspired algorithms based on collective behavior of decentralized, self-organized systems.

| Algorithm | Inspiration | Best For |
|-----------|-------------|----------|
| [Particle Swarm](./swarm-intelligence/particle-swarm) | Bird flocking | General-purpose, fast convergence |
| [Ant Colony](./swarm-intelligence/ant-colony) | Ant behavior | Discrete/continuous optimization |
| [Firefly Algorithm](./swarm-intelligence/firefly) | Firefly flashing | Multi-modal problems |
| [Grey Wolf](./swarm-intelligence/grey-wolf) | Wolf pack hunting | Exploration-exploitation balance |
| [Whale Optimization](./swarm-intelligence/whale) | Humpback whales | Large-scale problems |
| [Cuckoo Search](./swarm-intelligence/cuckoo) | Cuckoo birds | Global optimization |

### 🧬 Evolutionary (6 algorithms)

Algorithms inspired by biological evolution and natural selection.

| Algorithm | Key Feature | Best For |
|-----------|-------------|----------|
| [Genetic Algorithm](./evolutionary/genetic-algorithm) | Crossover, mutation | Discrete and continuous |
| [Differential Evolution](./evolutionary/differential-evolution) | Vector differences | Robust global search |
| [CMA-ES](./evolutionary/cma-es) | Covariance adaptation | High-dimensional |
| Cultural Algorithm | Belief space | Knowledge-based optimization |

### 🧠 Gradient-Based (11 algorithms)

Optimizers using gradient information for smooth landscapes.

| Algorithm | Key Feature | Best For |
|-----------|-------------|----------|
| [SGD Momentum](./gradient-based/sgd-momentum) | Momentum term | Simple problems |
| [Adam](./gradient-based/adam) | Adaptive moments | Deep learning |
| [AdamW](./gradient-based/adamw) | Weight decay | Regularized optimization |
| [RMSprop](./gradient-based/rmsprop) | RMS scaling | Non-stationary |

### 🎯 Classical (9 algorithms)

Traditional mathematical optimization methods.

| Algorithm | Type | Best For |
|-----------|------|----------|
| [BFGS](./classical/bfgs) | Quasi-Newton | Smooth functions |
| [Nelder-Mead](./classical/nelder-mead) | Direct search | Derivative-free |
| [Simulated Annealing](./classical/simulated-annealing) | Probabilistic | Global optimization |
| Hill Climbing | Local search | Unimodal problems |

### 🔬 Metaheuristic (12 algorithms)

High-level problem-independent algorithmic frameworks.

| Algorithm | Inspiration | Best For |
|-----------|-------------|----------|
| [Harmony Search](./metaheuristic/harmony-search) | Music improvisation | Discrete/continuous |
| Cross Entropy | Information theory | Rare event simulation |
| Sine Cosine | Mathematical functions | Multi-modal |

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
