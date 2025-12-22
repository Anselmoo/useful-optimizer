---
layout: home

hero:
  name: "Useful Optimizer"
  text: "54+ Optimization Algorithms"
  tagline: A comprehensive Python library for numeric optimization problems
  image:
    src: /logo.svg
    alt: Useful Optimizer
  actions:
    - theme: brand
      text: Get Started
      link: /guide/
    - theme: alt
      text: View on GitHub
      link: https://github.com/Anselmoo/useful-optimizer

features:
  - icon: 🦋
    title: Swarm Intelligence
    details: 57+ nature-inspired algorithms including Particle Swarm, Ant Colony, Firefly, Grey Wolf, and Whale Optimization.
  - icon: 🧬
    title: Evolutionary Algorithms
    details: Genetic Algorithm, Differential Evolution, CMA-ES, and Cultural Algorithm for robust global optimization.
  - icon: 🧠
    title: Gradient-Based
    details: 11 optimizers including Adam, AdamW, RMSprop, and Nesterov Accelerated Gradient for smooth landscapes.
  - icon: 🎯
    title: Classical Methods
    details: BFGS, Nelder-Mead, Simulated Annealing, and Trust Region methods with proven convergence.
  - icon: 🔬
    title: Metaheuristic
    details: Harmony Search, Cross Entropy, Sine Cosine, and Variable Neighbourhood Search algorithms.
  - icon: 📊
    title: Scientific Benchmarks
    details: Research-grade visualization with ECDF curves, convergence plots, and statistical comparison.
---

<script setup>
import { VPTeamMembers } from 'vitepress/theme'
</script>

## Quick Example

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

# Create optimizer
optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-12.768,
    upper_bound=12.768,
    dim=2,
    max_iter=100
)

# Run optimization
best_solution, best_fitness = optimizer.search()
print(f"Best fitness: {best_fitness:.6f}")
```

## Algorithm Categories

<div class="feature-grid">
  <div class="feature-item">
    <h4>🦋 Swarm Intelligence</h4>
    <p>Particle Swarm, Ant Colony, Firefly, Bat, Grey Wolf, Whale Optimization, and 50+ more nature-inspired algorithms.</p>
  </div>
  <div class="feature-item">
    <h4>🧬 Evolutionary</h4>
    <p>Genetic Algorithm, Differential Evolution, CMA-ES, Cultural Algorithm, and Imperialist Competitive Algorithm.</p>
  </div>
  <div class="feature-item">
    <h4>🧠 Gradient-Based</h4>
    <p>SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, AMSGrad, and Nesterov Accelerated Gradient.</p>
  </div>
  <div class="feature-item">
    <h4>🎯 Classical</h4>
    <p>BFGS, L-BFGS, Nelder-Mead, Powell, Simulated Annealing, Hill Climbing, and Trust Region methods.</p>
  </div>
</div>

## Installation

::: code-group

```bash [pip]
pip install git+https://github.com/Anselmoo/useful-optimizer
```

```bash [uv]
uv add git+https://github.com/Anselmoo/useful-optimizer
```

:::
