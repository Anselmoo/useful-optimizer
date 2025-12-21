# Social-Inspired Algorithms

Social-inspired optimization algorithms model human social interactions and group dynamics to solve optimization problems.

## Overview

| Property | Value |
|----------|-------|
| **Category** | Social Behavior |
| **Algorithms** | 4 |
| **Best For** | Population-based optimization |
| **Characteristic** | Human behavior modeling |

## Algorithm List

### Social Group Optimization

Based on human social group behavior and knowledge sharing.

```python
from opt.social_inspired import SocialGroupOptimizer

optimizer = SocialGroupOptimizer(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,
    max_iter=500,
)
```

### Teaching-Learning-Based Optimization

Inspired by the teaching-learning process in a classroom.

```python
from opt.social_inspired import TeachingLearning

optimizer = TeachingLearning(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,
    max_iter=500,
)
```

### Complete Algorithm List

| Algorithm | Social Phenomenon | Module |
|-----------|-------------------|--------|
| Political Optimizer | Political campaigns | `political_optimizer` |
| Soccer League Optimizer | Soccer competitions | `soccer_league_optimizer` |
| Social Group Optimizer | Social groups | `social_group_optimizer` |
| Teaching-Learning | Classroom learning | `teaching_learning` |

## See Also

- [API Reference: Social-Inspired](../api/social-inspired.md)
