# Metaheuristic Algorithms

Metaheuristics are high-level problem-independent algorithmic frameworks that provide a set of guidelines or strategies to develop heuristic optimization algorithms.

## Overview

| Property | Value |
|----------|-------|
| **Category** | Problem-Independent Frameworks |
| **Algorithms** | 14 |
| **Best For** | Complex optimization landscapes |
| **Flexibility** | High adaptability |

## Algorithm List

### Harmony Search

A music-inspired algorithm mimicking the improvisation process of musicians.

```python
from opt.metaheuristic import HarmonySearch

optimizer = HarmonySearch(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=30,
    max_iter=500,
)
```

### Cross Entropy Method

A Monte Carlo method for rare event simulation and optimization.

```python
from opt.metaheuristic import CrossEntropyMethod

optimizer = CrossEntropyMethod(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=100,
    max_iter=500,
)
```

### Complete Algorithm List

| Algorithm | Inspiration | Module |
|-----------|-------------|--------|
| Arithmetic Optimization | Mathematical operations | `arithmetic_optimization` |
| Colliding Bodies | Physics collision | `colliding_bodies_optimization` |
| Cross Entropy Method | Statistical sampling | `cross_entropy_method` |
| Eagle Strategy | Eagle hunting | `eagle_strategy` |
| Forensic-Based Investigation | Crime investigation | `forensic_based` |
| Harmony Search | Musical improvisation | `harmony_search` |
| Particle Filter | Bayesian filtering | `particle_filter` |
| Shuffled Frog Leaping | Frog behavior | `shuffled_frog_leaping_algorithm` |
| Sine Cosine Algorithm | Mathematical functions | `sine_cosine_algorithm` |
| Stochastic Diffusion Search | Information diffusion | `stochastic_diffusion_search` |
| Stochastic Fractal Search | Fractal geometry | `stochastic_fractal_search` |
| Variable Depth Search | Search depth variation | `variable_depth_search` |
| Variable Neighbourhood Search | Neighbourhood exploration | `variable_neighbourhood_search` |
| Very Large Scale Neighborhood | Large neighbourhood | `very_large_scale_neighborhood_search` |

## See Also

- [API Reference: Metaheuristic](../api/metaheuristic.md)
