# Constants Module

This module centralizes magic numbers and default values used throughout the useful-optimizer library, improving code maintainability and reducing code quality violations.

## Purpose

The `opt/constants.py` module addresses the following goals:

1. **Eliminate Magic Numbers**: Replace hard-coded values with named constants for better readability
2. **Centralize Configuration**: Provide a single source of truth for algorithm parameters
3. **Improve Code Quality**: Reduce linting violations (PLR2004 - Magic value comparisons)
4. **Enhance Documentation**: Each constant includes a docstring explaining its purpose

## Organization

Constants are organized into the following categories:

### Population and Iteration Defaults
- `DEFAULT_POPULATION_SIZE` - Default population size for population-based algorithms
- `DEFAULT_MAX_ITERATIONS` - Default maximum number of iterations
- `DEFAULT_SEED` - Default random seed for reproducibility

### Convergence and Tolerance Thresholds
- `DEFAULT_TOLERANCE` - General convergence tolerance
- `DEFAULT_CONVERGENCE_THRESHOLD` - Stricter convergence threshold
- `BARRIER_METHOD_MIN_MU` - Minimum barrier parameter
- `PENALTY_METHOD_TOLERANCE` - Penalty method tolerance

### Numerical Stability Constants
- `EPSILON_STABILITY` - Small constant for numerical stability
- `EPSILON_GRADIENT` - Epsilon for gradient computation
- `ADAM_EPSILON` - Epsilon for Adam-family optimizers

### Particle Swarm Optimization (PSO) Constants
- `PSO_INERTIA_WEIGHT` - Inertia weight for velocity updates
- `PSO_COGNITIVE_COEFFICIENT` - Cognitive coefficient (c1)
- `PSO_SOCIAL_COEFFICIENT` - Social coefficient (c2)
- `PSO_CONSTRICTION_COEFFICIENT` - Constriction coefficient
- `PSO_ACCELERATION_COEFFICIENT` - Acceleration coefficient

### Gradient-Based Optimizer Constants
- `ADAM_BETA1` - First moment decay for Adam-family optimizers
- `ADAM_BETA2` - Second moment decay for Adam-family optimizers
- `ADAMW_LEARNING_RATE` - Default learning rate for AdamW
- `NADAM_LEARNING_RATE` - Default learning rate for Nadam
- `SGD_MOMENTUM` - Momentum coefficient for SGD
- `RMSPROP_DECAY_RATE` - Decay rate for RMSprop

### Algorithm-Specific Constants
- `GOLDEN_RATIO` - The golden ratio (φ ≈ 1.618)
- `ELITE_FRACTION` - Elite sample fraction in cross-entropy method
- `TRAINING_PROBABILITY` - Training phase probability
- `BANDWIDTH_DEFAULT` - Default KDE bandwidth
- `GAMMA_DEFAULT` - Default gamma parameter

### Benchmark Function Constants
- `ACKLEY_A`, `ACKLEY_B`, `ACKLEY_C` - Ackley function constants
- `MATYAS_COEFFICIENT_A`, `MATYAS_COEFFICIENT_B` - Matyas function coefficients

### Bound Defaults
- `SHIFTED_ACKLEY_BOUND` - Typical bound for shifted Ackley function
- `ACKLEY_BOUND` - Typical bound for standard Ackley function
- `SPHERE_BOUND` - Typical bound for sphere function
- `ROSENBROCK_BOUND` - Typical bound for Rosenbrock function

## Usage

### In Optimizer Implementations

```python
from opt.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_POPULATION_SIZE,
    PSO_COGNITIVE_COEFFICIENT,
    PSO_INERTIA_WEIGHT,
    PSO_SOCIAL_COEFFICIENT,
)

class ParticleSwarm(AbstractOptimizer):
    def __init__(
        self,
        func: Callable,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = DEFAULT_POPULATION_SIZE,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
        c1: float = PSO_COGNITIVE_COEFFICIENT,
        c2: float = PSO_SOCIAL_COEFFICIENT,
        w: float = PSO_INERTIA_WEIGHT,
        # ... other parameters
    ) -> None:
        # Implementation
```

### In Test Code

```python
from opt.constants import (
    SHIFTED_ACKLEY_BOUND,
    DEFAULT_MAX_ITERATIONS,
)

def test_optimizer():
    optimizer = SomeOptimizer(
        func=shifted_ackley,
        lower_bound=-SHIFTED_ACKLEY_BOUND,
        upper_bound=SHIFTED_ACKLEY_BOUND,
        dim=2,
        max_iter=DEFAULT_MAX_ITERATIONS,
    )
```

## Migration Status

### Completed ✅
- [x] `opt/abstract_optimizer.py` - Base optimizer class
- [x] `opt/swarm_intelligence/particle_swarm.py` - PSO algorithm
- [x] `opt/gradient_based/nadam.py` - Nadam optimizer
- [x] `opt/gradient_based/adamw.py` - AdamW optimizer
- [x] Test suite for constants module

### In Progress 🚧
- [ ] Remaining gradient-based optimizers (adadelta, adagrad, adamax, amsgrad, rmsprop, sgd_momentum)
- [ ] Swarm intelligence algorithms (ant_colony, bat_algorithm, firefly, etc.)
- [ ] Evolutionary algorithms
- [ ] Physics-inspired algorithms
- [ ] Social-inspired algorithms
- [ ] Multi-objective algorithms
- [ ] Constrained optimization algorithms

### Migration Strategy

The migration is being done incrementally to:
1. Minimize disruption to the codebase
2. Allow thorough testing of each change
3. Maintain backward compatibility
4. Gradually reduce PLR2004 linting violations

## Benefits

1. **Code Readability**: `PSO_INERTIA_WEIGHT` is more readable than `0.5`
2. **Maintainability**: Update a constant in one place rather than searching all files
3. **Consistency**: Ensures same default values across all algorithms
4. **Documentation**: Each constant has a docstring explaining its purpose
5. **Testing**: Constants can be unit tested for correctness
6. **Code Quality**: Reduces linting violations and technical debt

## References

Constants are based on:
- Published literature (citations in algorithm implementations)
- Common practice in optimization research
- Empirical testing and validation
- Numerical stability requirements

## Contributing

When adding new optimization algorithms or updating existing ones:

1. Check if needed constants already exist in `opt/constants.py`
2. If not, add new constants with descriptive names and docstrings
3. Group related constants together
4. Update this README with the new constants
5. Add tests to `opt/test/test_constants.py`

## Testing

The constants module includes comprehensive tests:

```bash
uv run pytest opt/test/test_constants.py -v
```

Tests verify:
- Constants have expected types
- Constants are in valid ranges
- Relationships between constants are maintained
- No unintended zero values
- Mathematical relationships (e.g., golden ratio)
