# Swarm Intelligence Algorithms

Swarm intelligence algorithms are inspired by the collective behavior of decentralized, self-organized systems in nature. These algorithms mimic the social behavior of birds, fish, insects, and other animals that exhibit collective intelligence.

## Overview

| Property | Value |
|----------|-------|
| **Category** | Nature-Inspired |
| **Algorithms** | 57 |
| **Best For** | Global optimization, multimodal functions |
| **Typical Population Size** | 20-100 |

## Algorithm List

### Particle Swarm Optimization (PSO)

The most well-known swarm algorithm, inspired by bird flocking and fish schooling.

```python
from opt.swarm_intelligence import ParticleSwarm

optimizer = ParticleSwarm(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,
    max_iter=500,
    c1=2.0,  # Cognitive coefficient
    c2=2.0,  # Social coefficient
)
```

### Grey Wolf Optimizer (GWO)

Mimics the leadership hierarchy and hunting mechanism of grey wolves.

```python
from opt.swarm_intelligence import GreyWolfOptimizer

optimizer = GreyWolfOptimizer(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=30,
    max_iter=500,
)
```

### Ant Colony Optimization (ACO)

Based on the foraging behavior of ants finding paths between colony and food.

```python
from opt.swarm_intelligence import AntColony

optimizer = AntColony(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=50,
    max_iter=500,
)
```

### Whale Optimization Algorithm (WOA)

Inspired by the bubble-net hunting strategy of humpback whales.

```python
from opt.swarm_intelligence import WhaleOptimizationAlgorithm

optimizer = WhaleOptimizationAlgorithm(
    func=objective,
    lower_bound=-5,
    upper_bound=5,
    dim=10,
    population_size=30,
    max_iter=500,
)
```

### Complete Algorithm List

<div class="annotate" markdown>

| Algorithm | Inspiration | Module |
|-----------|-------------|--------|
| African Buffalo Optimization | Buffalo herd behavior | `african_buffalo_optimization` |
| African Vultures Optimizer | Vulture foraging | `african_vultures_optimizer` |
| Ant Colony Optimization | Ant foraging | `ant_colony` |
| Ant Lion Optimizer | Antlion hunting | `ant_lion_optimizer` |
| Aquila Optimizer | Eagle hunting | `aquila_optimizer` |
| Artificial Fish Swarm | Fish schooling | `artificial_fish_swarm_algorithm` |
| Artificial Gorilla Troops | Gorilla behavior | `artificial_gorilla_troops` |
| Artificial Hummingbird | Hummingbird foraging | `artificial_hummingbird` |
| Artificial Rabbits | Rabbit survival | `artificial_rabbits` |
| Barnacles Mating Optimizer | Barnacle reproduction | `barnacles_mating` |
| Bat Algorithm | Bat echolocation | `bat_algorithm` |
| Bee Algorithm | Bee foraging | `bee_algorithm` |
| Black Widow Optimization | Spider behavior | `black_widow` |
| Brown Bear Optimization | Bear behavior | `brown_bear` |
| Cat Swarm Optimization | Cat behavior | `cat_swarm_optimization` |
| Chimp Optimization | Chimpanzee behavior | `chimp_optimization` |
| Coati Optimization | Coati foraging | `coati_optimizer` |
| Cuckoo Search | Cuckoo parasitism | `cuckoo_search` |
| Dandelion Optimizer | Dandelion seed dispersal | `dandelion_optimizer` |
| Dingo Optimizer | Dingo hunting | `dingo_optimizer` |
| Dragonfly Algorithm | Dragonfly swarming | `dragonfly_algorithm` |
| Emperor Penguin Optimizer | Penguin huddling | `emperor_penguin` |
| Fennec Fox Optimization | Fox survival | `fennec_fox` |
| Firefly Algorithm | Firefly flashing | `firefly_algorithm` |
| Flower Pollination | Plant pollination | `flower_pollination` |
| Giant Trevally Optimizer | Fish hunting | `giant_trevally` |
| Glowworm Swarm | Glowworm behavior | `glowworm_swarm_optimization` |
| Golden Eagle Optimizer | Eagle hunting | `golden_eagle` |
| Grasshopper Optimization | Grasshopper swarming | `grasshopper_optimization` |
| Grey Wolf Optimizer | Wolf pack hierarchy | `grey_wolf_optimizer` |
| Harris Hawks Optimization | Hawks hunting | `harris_hawks_optimization` |
| Honey Badger Algorithm | Badger behavior | `honey_badger` |
| Manta Ray Foraging | Manta ray feeding | `manta_ray` |
| Marine Predators Algorithm | Ocean predator behavior | `marine_predators_algorithm` |
| Mayfly Algorithm | Mayfly mating | `mayfly_optimizer` |
| Moth-Flame Optimization | Moth navigation | `moth_flame_optimization` |
| Moth Search | Moth behavior | `moth_search` |
| Mountain Gazelle Optimizer | Gazelle behavior | `mountain_gazelle` |
| Orca Predator Algorithm | Orca hunting | `orca_predator` |
| Osprey Optimization | Osprey hunting | `osprey_optimizer` |
| Particle Swarm Optimization | Bird flocking | `particle_swarm` |
| Pathfinder Algorithm | Group navigation | `pathfinder` |
| Pelican Optimization | Pelican hunting | `pelican_optimizer` |
| Reptile Search | Reptile hunting | `reptile_search` |
| Salp Swarm Algorithm | Salp chains | `salp_swarm_algorithm` |
| Sand Cat Swarm | Sand cat hunting | `sand_cat` |
| Seagull Optimization | Seagull migration | `seagull_optimization` |
| Slime Mould Algorithm | Slime mould behavior | `slime_mould` |
| Snow Geese Algorithm | Geese migration | `snow_geese` |
| Spotted Hyena Optimizer | Hyena hunting | `spotted_hyena` |
| Squirrel Search | Squirrel foraging | `squirrel_search` |
| Starling Murmuration | Starling flocking | `starling_murmuration` |
| Tunicate Swarm | Tunicate behavior | `tunicate_swarm` |
| Whale Optimization | Whale hunting | `whale_optimization_algorithm` |
| Wild Horse Optimizer | Horse herding | `wild_horse` |
| Zebra Optimization | Zebra behavior | `zebra_optimizer` |

</div>

## Usage Example

```python
from opt.swarm_intelligence import (
    ParticleSwarm,
    GreyWolfOptimizer,
    WhaleOptimizationAlgorithm,
    AntColony,
    FireflyAlgorithm,
)
from opt.benchmark.functions import rosenbrock

# Compare multiple swarm algorithms
results = {}
for OptimizerClass in [ParticleSwarm, GreyWolfOptimizer, WhaleOptimizationAlgorithm]:
    optimizer = OptimizerClass(
        func=rosenbrock,
        lower_bound=-5,
        upper_bound=5,
        dim=10,
        max_iter=500,
    )
    _, fitness = optimizer.search()
    results[OptimizerClass.__name__] = fitness

for name, fitness in results.items():
    print(f"{name}: {fitness:.6e}")
```

## See Also

- [API Reference: Swarm Intelligence](../api/swarm-intelligence.md)
- [Benchmark Results](../benchmarks/results.md)
