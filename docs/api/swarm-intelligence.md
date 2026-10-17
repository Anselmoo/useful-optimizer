# Swarm Intelligence API

API reference for the 56 classes of `opt.swarm_intelligence`. This page is generated from the source; see the [algorithm pages](/algorithms/swarm-intelligence/) for formulations and references.

## Imports

```python
from opt.swarm_intelligence import (
    AfricanBuffaloOptimizer,
    AfricanVulturesOptimizer,
    AntColony,
    AntLionOptimizer,
    AquilaOptimizer,
    ArtificialFishSwarm,
    ArtificialGorillaTroopsOptimizer,
    ArtificialHummingbirdAlgorithm,
    ArtificialRabbitsOptimizer,
    BarnaclesMatingOptimizer,
    BatAlgorithm,
    BeeAlgorithm,
    BlackWidowOptimizer,
    BrownBearOptimizer,
    CatSwarmOptimization,
    ChimpOptimizationAlgorithm,
    CoatiOptimizer,
    CuckooSearch,
    DandelionOptimizer,
    DingoOptimizer,
    DragonflyOptimizer,
    EmperorPenguinOptimizer,
    FennecFoxOptimizer,
    FireflyAlgorithm,
    FlowerPollinationAlgorithm,
    GiantTrevallyOptimizer,
    GlowwormSwarmOptimization,
    GoldenEagleOptimizer,
    GrasshopperOptimizer,
    GreyWolfOptimizer,
    HarrisHawksOptimizer,
    HoneyBadgerAlgorithm,
    MantaRayForagingOptimization,
    MarinePredatorsOptimizer,
    MayflyOptimizer,
    MothSearchAlgorithm,
    MothFlameOptimizer,
    MountainGazelleOptimizer,
    OrcaPredatorAlgorithm,
    OspreyOptimizer,
    ParticleSwarm,
    PathfinderAlgorithm,
    PelicanOptimizer,
    ReptileSearchAlgorithm,
    SalpSwarmOptimizer,
    SandCatSwarmOptimizer,
    SeagullOptimizationAlgorithm,
    SlimeMouldAlgorithm,
    SnowGeeseOptimizer,
    SpottedHyenaOptimizer,
    SquirrelSearchAlgorithm,
    StarlingMurmurationOptimizer,
    TunicateSwarmAlgorithm,
    WhaleOptimizationAlgorithm,
    WildHorseOptimizer,
    ZebraOptimizer,
)
```

## Common interface

Every class takes the objective `func`, the search-space `lower_bound`, `upper_bound` and `dim`, usually `max_iter` and `seed`, and implements `search() -> tuple[np.ndarray, float]`. See [AbstractOptimizer](./abstract-optimizer).

## Classes

| Class | Algorithm | Additional parameters |
|-------|-----------|-----------------------|
| [`AfricanBuffaloOptimizer`](/algorithms/swarm-intelligence/african-buffalo-optimization) | African Buffalo Optimization | `lp1=_LP1`, `lp2=_LP2` |
| [`AfricanVulturesOptimizer`](/algorithms/swarm-intelligence/african-vultures-optimizer) | African Vultures Optimization Algorithm | - |
| [`AntColony`](/algorithms/swarm-intelligence/ant-colony) | Ant Colony Optimization | `alpha=1`, `beta=1`, `rho=0.5`, `q=1` |
| [`AntLionOptimizer`](/algorithms/swarm-intelligence/ant-lion-optimizer) | Ant Lion Optimizer | - |
| [`AquilaOptimizer`](/algorithms/swarm-intelligence/aquila-optimizer) | Aquila Optimizer | - |
| [`ArtificialFishSwarm`](/algorithms/swarm-intelligence/artificial-fish-swarm-algorithm) | Artificial Fish Swarm Algorithm | `fish_swarm=50`, `visual=1`, `step=0.1`, `try_number=3`, `epsilon=1e-09` |
| [`ArtificialGorillaTroopsOptimizer`](/algorithms/swarm-intelligence/artificial-gorilla-troops) | Artificial Gorilla Troops Optimizer | - |
| [`ArtificialHummingbirdAlgorithm`](/algorithms/swarm-intelligence/artificial-hummingbird) | Artificial Hummingbird Algorithm | - |
| [`ArtificialRabbitsOptimizer`](/algorithms/swarm-intelligence/artificial-rabbits) | Artificial Rabbits Optimization | - |
| [`BarnaclesMatingOptimizer`](/algorithms/swarm-intelligence/barnacles-mating) | Barnacles Mating Optimizer | `pl=_PL` |
| [`BatAlgorithm`](/algorithms/swarm-intelligence/bat-algorithm) | Bat Algorithm | `n_bats`, `loudness=0.5`, `pulse_rate=0.9`, `freq_min=0`, `freq_max=2` |
| [`BeeAlgorithm`](/algorithms/swarm-intelligence/bee-algorithm) | Bee Algorithm | `n_bees=50`, `scout_bee=0.01` |
| [`BlackWidowOptimizer`](/algorithms/swarm-intelligence/black-widow) | Black Widow Optimization Algorithm | `pp=_PP`, `cr=_CR`, `pm=_PM` |
| [`BrownBearOptimizer`](/algorithms/swarm-intelligence/brown-bear) | Brown Bear Optimization Algorithm | - |
| [`CatSwarmOptimization`](/algorithms/swarm-intelligence/cat-swarm-optimization) | Cat Swarm Optimization | `cats=50`, `seeking_memory_pool=5`, `counts_of_dimension_to_change=None`, `smp_change_probability=0.1`, `spc_probability=0.2` |
| [`ChimpOptimizationAlgorithm`](/algorithms/swarm-intelligence/chimp-optimization) | Chimp Optimization Algorithm | - |
| [`CoatiOptimizer`](/algorithms/swarm-intelligence/coati-optimizer) | Coati Optimization Algorithm | - |
| [`CuckooSearch`](/algorithms/swarm-intelligence/cuckoo-search) | Cuckoo Search | `mutation_probability=0.1` |
| [`DandelionOptimizer`](/algorithms/swarm-intelligence/dandelion-optimizer) | Dandelion Optimizer | - |
| [`DingoOptimizer`](/algorithms/swarm-intelligence/dingo-optimizer) | Dingo Optimization Algorithm | `survival_rate=_SURVIVAL_RATE` |
| [`DragonflyOptimizer`](/algorithms/swarm-intelligence/dragonfly-algorithm) | Dragonfly Algorithm | - |
| [`EmperorPenguinOptimizer`](/algorithms/swarm-intelligence/emperor-penguin) | Emperor Penguin Optimizer | - |
| [`FennecFoxOptimizer`](/algorithms/swarm-intelligence/fennec-fox) | Fennec Fox Optimization Algorithm | - |
| [`FireflyAlgorithm`](/algorithms/swarm-intelligence/firefly-algorithm) | Firefly Algorithm | `alpha=0.5`, `beta_0=1`, `gamma=1` |
| [`FlowerPollinationAlgorithm`](/algorithms/swarm-intelligence/flower-pollination) | Flower Pollination Algorithm | `switch_probability=_SWITCH_PROBABILITY` |
| [`GiantTrevallyOptimizer`](/algorithms/swarm-intelligence/giant-trevally) | Giant Trevally Optimizer | - |
| [`GlowwormSwarmOptimization`](/algorithms/swarm-intelligence/glowworm-swarm-optimization) | Glowworm Swarm Optimization | `luciferin_decay=0.1`, `randomness=0.5`, `step_size=0.01` |
| [`GoldenEagleOptimizer`](/algorithms/swarm-intelligence/golden-eagle) | Golden Eagle Optimizer | - |
| [`GrasshopperOptimizer`](/algorithms/swarm-intelligence/grasshopper-optimization) | Grasshopper Optimization Algorithm | `c_max=_C_MAX`, `c_min=_C_MIN`, `f=_ATTRACTION_INTENSITY`, `l=_ATTRACTIVE_LENGTH_SCALE` |
| [`GreyWolfOptimizer`](/algorithms/swarm-intelligence/grey-wolf-optimizer) | Grey Wolf Optimizer | - |
| [`HarrisHawksOptimizer`](/algorithms/swarm-intelligence/harris-hawks-optimization) | Harris Hawks Optimization | - |
| [`HoneyBadgerAlgorithm`](/algorithms/swarm-intelligence/honey-badger) | Honey Badger Algorithm | `beta=_BETA` |
| [`MantaRayForagingOptimization`](/algorithms/swarm-intelligence/manta-ray) | Manta Ray Foraging Optimization | - |
| [`MarinePredatorsOptimizer`](/algorithms/swarm-intelligence/marine-predators-algorithm) | Marine Predators Algorithm | `fads=_FADs_EFFECT_PROB` |
| [`MayflyOptimizer`](/algorithms/swarm-intelligence/mayfly-optimizer) | Mayfly Algorithm | `a1=_A1`, `a2=_A2`, `a3=_A3`, `beta=_BETA`, `dance=_DANCE`, `fl=_FL`, `g=_G` |
| [`MothSearchAlgorithm`](/algorithms/swarm-intelligence/moth-search) | Moth Search Algorithm | `path_finder_ratio=0.5` |
| [`MothFlameOptimizer`](/algorithms/swarm-intelligence/moth-flame-optimization) | Moth-Flame Optimization Algorithm | `b=1.0` |
| [`MountainGazelleOptimizer`](/algorithms/swarm-intelligence/mountain-gazelle) | Mountain Gazelle Optimizer | - |
| [`OrcaPredatorAlgorithm`](/algorithms/swarm-intelligence/orca-predator) | Orca Predation Algorithm | - |
| [`OspreyOptimizer`](/algorithms/swarm-intelligence/osprey-optimizer) | Osprey Optimization Algorithm | - |
| [`ParticleSwarm`](/algorithms/swarm-intelligence/particle-swarm) | Particle Swarm Optimization | `c1=PSO_COGNITIVE_COEFFICIENT`, `c2=PSO_SOCIAL_COEFFICIENT`, `w=PSO_INERTIA_WEIGHT` |
| [`PathfinderAlgorithm`](/algorithms/swarm-intelligence/pathfinder) | Pathfinder Algorithm | - |
| [`PelicanOptimizer`](/algorithms/swarm-intelligence/pelican-optimizer) | Pelican Optimization Algorithm | - |
| [`ReptileSearchAlgorithm`](/algorithms/swarm-intelligence/reptile-search) | Reptile Search Algorithm | - |
| [`SalpSwarmOptimizer`](/algorithms/swarm-intelligence/salp-swarm-algorithm) | Salp Swarm Algorithm | - |
| [`SandCatSwarmOptimizer`](/algorithms/swarm-intelligence/sand-cat) | Sand Cat Swarm Optimization | - |
| [`SeagullOptimizationAlgorithm`](/algorithms/swarm-intelligence/seagull-optimization) | Seagull Optimization Algorithm | - |
| [`SlimeMouldAlgorithm`](/algorithms/swarm-intelligence/slime-mould) | Slime Mould Algorithm | - |
| [`SnowGeeseOptimizer`](/algorithms/swarm-intelligence/snow-geese) | Snow Geese Algorithm | - |
| [`SpottedHyenaOptimizer`](/algorithms/swarm-intelligence/spotted-hyena) | Spotted Hyena Optimizer | - |
| [`SquirrelSearchAlgorithm`](/algorithms/swarm-intelligence/squirrel-search) | Squirrel Search Algorithm | - |
| [`StarlingMurmurationOptimizer`](/algorithms/swarm-intelligence/starling-murmuration) | Starling Murmuration Optimizer | - |
| [`TunicateSwarmAlgorithm`](/algorithms/swarm-intelligence/tunicate-swarm) | Tunicate Swarm Algorithm | - |
| [`WhaleOptimizationAlgorithm`](/algorithms/swarm-intelligence/whale-optimization-algorithm) | Whale Optimization Algorithm | - |
| [`WildHorseOptimizer`](/algorithms/swarm-intelligence/wild-horse) | Wild Horse Optimizer | `n_groups=5` |
| [`ZebraOptimizer`](/algorithms/swarm-intelligence/zebra-optimizer) | Zebra Optimization Algorithm | - |
