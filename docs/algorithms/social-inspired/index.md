# Social-Inspired Algorithms

Optimization algorithms based on human social behavior and interactions.

## Overview

Social-inspired algorithms model human social phenomena such as teaching, learning, cooperation, competition, and cultural evolution to solve optimization problems.

## Available Algorithms

- [Teaching-Learning Based Optimization](./tlbo) - Simulates teaching and learning in a classroom
- [Social Spider Optimization](./social-spider) - Models spider colony behavior
- [League Championship Algorithm](./league-championship) - Sports competition simulation
- [Soccer League Optimization](./soccer-league) - Football league dynamics

## Characteristics

- Model social interactions and behaviors
- Use cooperation and competition mechanisms
- Often involve knowledge sharing
- Can incorporate learning and adaptation

## Usage Example

```python
from opt.social_inspired import TeachingLearning
from opt.benchmark.functions import rastrigin

optimizer = TeachingLearning(
    func=rastrigin,
    lower_bound=-5.12,
    upper_bound=5.12,
    dim=10,
    max_iter=100,
    population_size=30
)
best_solution, best_fitness = optimizer.search()
```

## See Also

- [API Reference](/api/) - Complete API documentation
- [Swarm Intelligence](../swarm-intelligence/) - Related collective behavior algorithms
