---
layout: home

hero:
  name: useful-optimizer
  text: Optimization Algorithms Library
  tagline: 120+ optimization algorithms for numeric problems
  actions:
    - theme: brand
      text: Get Started
      link: /guide/installation
    - theme: alt
      text: View on GitHub
      link: https://github.com/Anselmoo/useful-optimizer

features:
  - icon: 🐝
    title: Swarm Intelligence
    details: 56 nature-inspired algorithms including PSO, ACO, and more
  - icon: 🧬
    title: Evolutionary
    details: Genetic algorithms, differential evolution, CMA-ES
  - icon: 📊
    title: Scientific Benchmarks
    details: COCO/BBOB compliant benchmark suite with 30-run statistics
  - icon: 📈
    title: Interactive Visualizations
    details: ECharts-powered ECDF curves, convergence plots, 3D landscapes
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
