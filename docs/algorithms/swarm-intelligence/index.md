# Swarm Intelligence Algorithms

Nature-inspired optimization algorithms based on collective behavior of decentralized, self-organized systems.

## Overview

Swarm intelligence algorithms mimic the collective behavior observed in nature, such as bird flocking, fish schooling, ant colonies, and bee swarms. These algorithms use multiple agents (particles, individuals) that interact locally with each other and their environment.

## Characteristics

- **Decentralized**: No central control mechanism
- **Self-organized**: Global behavior emerges from local interactions
- **Population-based**: Multiple agents work together
- **Stochastic**: Random components help explore solution space

## Available Algorithms

### Classic Swarm Algorithms

- [Particle Swarm Optimization (PSO)](./particle-swarm) - Inspired by bird flocking behavior
- [Ant Colony Optimization (ACO)](./ant-colony) - Based on ant foraging behavior
- [Artificial Bee Colony (ABC)](./bee) - Mimics honey bee foraging
- [Firefly Algorithm](./firefly) - Based on firefly flashing patterns
- [Bat Algorithm](./bat) - Inspired by bat echolocation

### Predator-Prey Algorithms

- [Grey Wolf Optimizer](./grey-wolf) - Simulates grey wolf hunting strategy
- [Whale Optimization](./whale) - Based on whale bubble-net feeding
- [Harris Hawks](./harris-hawks) - Cooperative hunting behavior
- [Marine Predators](./marine-predators) - Ocean predator hunting strategies

### And 50+ more algorithms!

See the sidebar for the complete list of available swarm intelligence algorithms.

## Usage Example

```python
from opt.swarm_intelligence import ParticleSwarm, AntColony, GreyWolf
from opt.benchmark.functions import shifted_ackley

# Particle Swarm Optimization
pso = ParticleSwarm(
    func=shifted_ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=10,
    max_iter=100,
    population_size=30
)
best_solution, best_fitness = pso.search()

# Ant Colony Optimization
aco = AntColony(
    func=shifted_ackley,
    lower_bound=-32.768,
    upper_bound=32.768,
    dim=10,
    max_iter=100
)
best_solution, best_fitness = aco.search()
```

## Common Parameters

Most swarm intelligence algorithms share these common parameters:

- `func`: Objective function to minimize
- `lower_bound`: Lower bounds for variables
- `upper_bound`: Upper bounds for variables
- `dim`: Problem dimension
- `max_iter`: Maximum number of iterations
- `population_size`: Number of agents/particles (algorithm-specific default)

## Performance Characteristics

Swarm intelligence algorithms generally:
- Work well for multi-modal problems
- Are robust to noise
- Can escape local optima
- Scale reasonably with problem dimension
- Are simple to implement and understand

## See Also

- [API Reference](/api/) - Complete API documentation
- [Benchmark Results](/benchmarks/results) - Algorithm performance comparisons
