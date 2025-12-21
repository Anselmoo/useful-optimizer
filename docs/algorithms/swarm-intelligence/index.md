---
title: Swarm Intelligence
description: Nature-inspired algorithms based on collective behavior
tags:
  - swarm-intelligence
  - population-based
  - metaheuristic
---

# Swarm Intelligence Algorithms

Swarm intelligence algorithms are inspired by the collective behavior of decentralized, self-organized systems, such as bird flocks, ant colonies, and fish schools.

---

## Overview

These algorithms leverage the emergent intelligence that arises from simple interactions between individuals in a population. Key characteristics include:

- **Decentralized control** - No central authority directing the search
- **Self-organization** - Global patterns emerge from local interactions
- **Population-based** - Multiple candidate solutions explore simultaneously
- **Adaptive** - Behavior changes based on feedback from the environment

---

## Algorithms

| Algorithm | Inspiration | Best For |
|-----------|-------------|----------|
| [**Particle Swarm**](particle-swarm.md) | Bird flocking | General optimization |
| [**Ant Colony**](ant-colony.md) | Ant foraging | Path finding, combinatorial |
| **Bat Algorithm** | Echolocation | Multimodal problems |
| **Bee Algorithm** | Honey bee foraging | Continuous optimization |
| **Cat Swarm** | Cat behavior | Continuous optimization |
| **Cuckoo Search** | Cuckoo breeding | Global optimization |
| **Firefly Algorithm** | Firefly flashing | Multimodal problems |
| **Fish Swarm** | Fish schooling | Global optimization |
| **Glowworm Swarm** | Glowworm behavior | Multimodal problems |
| **Grey Wolf** | Wolf hunting | Continuous optimization |
| **Squirrel Search** | Squirrel foraging | Dynamic optimization |
| **Whale Optimization** | Whale hunting | Continuous optimization |

---

## Common Parameters

All swarm intelligence algorithms share these parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `population_size` | `int` | Number of agents in the swarm |
| `max_iter` | `int` | Maximum number of iterations |
| `dim` | `int` | Problem dimensionality |
| `lower_bound` | `float` | Lower search boundary |
| `upper_bound` | `float` | Upper search boundary |

---

## Usage Pattern

```python
from opt.swarm_intelligence import ParticleSwarm
from opt.benchmark.functions import shifted_ackley

optimizer = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-10,
    upper_bound=10,
    dim=5,
    population_size=30,
    max_iter=100,
)

best_solution, best_fitness = optimizer.search()
```

---

## When to Use

!!! success "Good For"
    - Black-box optimization (no gradient information)
    - Multimodal landscapes with many local optima
    - Problems where exploration is important
    - Parallelizable evaluation

!!! warning "Limitations"
    - May require many function evaluations
    - Parameter tuning can affect performance
    - May be slower than gradient methods for smooth problems
