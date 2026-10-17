# Social-Inspired API

API reference for the 4 classes of `opt.social_inspired`. This page is generated from the source; see the [algorithm pages](/algorithms/social-inspired/) for formulations and references.

## Imports

```python
from opt.social_inspired import (
    PoliticalOptimizer,
    SoccerLeagueOptimizer,
    SocialGroupOptimizer,
    TeachingLearningOptimizer,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`PoliticalOptimizer`](/algorithms/social-inspired/political-optimizer) | Political Optimizer | `num_parties=5` |
| [`SoccerLeagueOptimizer`](/algorithms/social-inspired/soccer-league-optimizer) | Soccer League Competition | `num_teams=10` |
| [`SocialGroupOptimizer`](/algorithms/social-inspired/social-group-optimizer) | Social Group Optimization | `c=0.2`, `track_convergence=False`, `early_stopping=False`, `tolerance=1e-06`, `patience=10`, `verbose=False` |
| [`TeachingLearningOptimizer`](/algorithms/social-inspired/teaching-learning) | Teaching-Learning Based Optimization | - |
