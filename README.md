[![DOI](https://zenodo.org/badge/776526436.svg)](https://zenodo.org/doi/10.5281/zenodo.13294276)

# Useful Optimizer

Useful Optimizer is a dedicated set of optimization algorithms for numeric problems. It's designed to provide a comprehensive collection of optimization techniques that can be easily used and integrated into any project.

## Version

The current version of Useful Optimizer is 0.1.2.

## Features

- A wide range of optimization algorithms in ten categories, each documented and benchmarked.
- Organized into logical categories for easy discovery.
- Easy to use and integrate.
- Suitable for various numeric problems.
- Having fun to play with the algorithms

## Installation

To install Useful Optimizer, you can use pip:

```bash
pip install git+https://github.com/Anselmoo/useful-optimizer
```

Or using uv (recommended):

```bash
uv add git+https://github.com/Anselmoo/useful-optimizer
```

## Usage

Here's a basic example of how to use Useful Optimizer:

```python
from opt.metaheuristic import CrossEntropyMethod
from opt.benchmark.functions import shifted_ackley

optimizer = CrossEntropyMethod(
        func=shifted_ackley,
        dim=2,
        lower_bound=-12.768,
        upper_bound=+12.768,
        population_size=100,
        max_iter=1000,
    )
best_solution, best_fitness = optimizer.search()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

You can also use the gradient-based optimizers:

```python
from opt.gradient_based import SGD, AdamW
from opt.classical import BFGS
from opt.benchmark.functions import shifted_ackley

# Gradient-based optimization
sgd = SGD(func=shifted_ackley, lower_bound=-12.768, upper_bound=12.768, dim=2, learning_rate=0.01)
best_solution, best_fitness = sgd.search()

# Adam variant with weight decay
adamw = AdamW(func=shifted_ackley, lower_bound=-12.768, upper_bound=12.768, dim=2, weight_decay=0.01)
best_solution, best_fitness = adamw.search()

# Quasi-Newton method
bfgs = BFGS(func=shifted_ackley, lower_bound=-12.768, upper_bound=12.768, dim=2, num_restarts=10)
best_solution, best_fitness = bfgs.search()
```

### Import Styles

All optimizers can be imported in two ways:

```python
# Categorical imports (recommended for discoverability)
from opt.swarm_intelligence import ParticleSwarm
from opt.gradient_based import AdamW
from opt.classical import BFGS

# Direct imports from root (backward compatible)
from opt import ParticleSwarm, AdamW, BFGS
```

### Quick Demo

All optimizers include a standardized demo that can be run directly or customized:

```python
from opt.demo import run_demo
from opt.swarm_intelligence import ParticleSwarm

# Run with default settings
run_demo(ParticleSwarm)

# Or customize parameters
run_demo(
    ParticleSwarm,
    max_iter=200,
    population_size=50,
    c1=2.0,
    c2=2.0
)
```

You can also run demos directly from the command line:

```bash
python -m opt.swarm_intelligence.particle_swarm
python -m opt.gradient_based.adamw
python -m opt.classical.simulated_annealing
```

## Project Structure

Optimizers are organized into categorical subfolders:

<!-- BEGIN GENERATED: readme-structure (scripts/generate_docs.py) -->

```
src/opt/
├── abstract/            # Base classes and history tracking
├── swarm_intelligence/  # 56 algorithms
├── evolutionary/        # 6 algorithms
├── gradient_based/      # 11 algorithms
├── classical/           # 9 algorithms
├── metaheuristic/       # 14 algorithms
├── physics_inspired/    # 4 algorithms
├── probabilistic/       # 5 algorithms
├── social_inspired/     # 4 algorithms
├── constrained/         # 5 algorithms
├── multi_objective/     # 3 algorithms
└── benchmark/           # Benchmark functions
```

<!-- END GENERATED: readme-structure -->

## Implemented Optimizers

<!-- BEGIN GENERATED: readme-optimizers (scripts/generate_docs.py) -->

Useful Optimizer includes 117 optimization algorithms in 10 categories, each implemented as a separate module. Every optimizer links to its documentation page, which carries the formulation, parameters, references and benchmark charts, and to its source code.

<details>
<summary><strong>🦋 Swarm Intelligence (56)</strong></summary>

Nature-inspired algorithms based on collective behavior.

- **[African Buffalo Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/african-buffalo-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/african_buffalo_optimization.py)) - African Buffalo Optimization (ABO) optimization algorithm.
- **[African Vultures Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/african-vultures-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/african_vultures_optimizer.py)) - African Vultures Optimization Algorithm (AVOA) optimization algorithm.
- **[Ant Colony Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/ant-colony)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/ant_colony.py)) - Ant Colony Optimization (ACO) algorithm for continuous optimization.
- **[Ant Lion Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/ant-lion-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/ant_lion_optimizer.py)) - Ant Lion Optimizer (ALO) optimization algorithm.
- **[Aquila Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/aquila-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/aquila_optimizer.py)) - Aquila Optimizer (AO) optimization algorithm.
- **[Artificial Fish Swarm Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/artificial-fish-swarm-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/artificial_fish_swarm_algorithm.py)) - Artificial Fish Swarm Algorithm (AFSA) optimization algorithm.
- **[Artificial Gorilla Troops Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/artificial-gorilla-troops)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/artificial_gorilla_troops.py)) - Artificial Gorilla Troops Optimizer (GTO) optimization algorithm.
- **[Artificial Hummingbird Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/artificial-hummingbird)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/artificial_hummingbird.py)) - Artificial Hummingbird Algorithm (AHA) optimization algorithm.
- **[Artificial Rabbits Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/artificial-rabbits)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/artificial_rabbits.py)) - Artificial Rabbits Optimization (ARO) optimization algorithm.
- **[Barnacles Mating Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/barnacles-mating)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/barnacles_mating.py)) - Barnacles Mating Optimizer (BMO) optimization algorithm.
- **[Bat Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/bat-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/bat_algorithm.py)) - Bat Algorithm (BA) optimization algorithm.
- **[Bee Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/bee-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/bee_algorithm.py)) - Bee Algorithm (BA) optimization algorithm.
- **[Black Widow Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/black-widow)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/black_widow.py)) - Black Widow Optimization Algorithm (BWO) optimization algorithm.
- **[Brown Bear Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/brown-bear)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/brown_bear.py)) - Brown Bear Optimization Algorithm (BBOA) optimization algorithm.
- **[Cat Swarm Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/cat-swarm-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/cat_swarm_optimization.py)) - Cat Swarm Optimization (CSO) optimization algorithm.
- **[Chimp Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/chimp-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/chimp_optimization.py)) - Chimp Optimization Algorithm (ChOA) optimization algorithm.
- **[Coati Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/coati-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/coati_optimizer.py)) - Coati Optimization Algorithm (COA) optimization algorithm.
- **[Cuckoo Search](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/cuckoo-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/cuckoo_search.py)) - Cuckoo Search (CS) optimization algorithm.
- **[Dandelion Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/dandelion-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/dandelion_optimizer.py)) - Dandelion Optimizer (DO) optimization algorithm.
- **[Dingo Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/dingo-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/dingo_optimizer.py)) - Dingo Optimization Algorithm (DOA) optimization algorithm.
- **[Dragonfly Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/dragonfly-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/dragonfly_algorithm.py)) - Dragonfly Algorithm (DA) optimization algorithm.
- **[Emperor Penguin Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/emperor-penguin)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/emperor_penguin.py)) - Emperor Penguin Optimizer (EPO) optimization algorithm.
- **[Fennec Fox Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/fennec-fox)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/fennec_fox.py)) - Fennec Fox Optimization Algorithm (FFO) optimization algorithm.
- **[Firefly Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/firefly-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/firefly_algorithm.py)) - Firefly Algorithm (FA) optimization algorithm.
- **[Flower Pollination Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/flower-pollination)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/flower_pollination.py)) - Flower Pollination Algorithm (FPA) optimization algorithm.
- **[Giant Trevally Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/giant-trevally)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/giant_trevally.py)) - Giant Trevally Optimizer (GTO) optimization algorithm.
- **[Glowworm Swarm Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/glowworm-swarm-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/glowworm_swarm_optimization.py)) - Glowworm Swarm Optimization (GSO) optimization algorithm.
- **[Golden Eagle Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/golden-eagle)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/golden_eagle.py)) - Golden Eagle Optimizer (GEO) optimization algorithm.
- **[Grasshopper Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/grasshopper-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/grasshopper_optimization.py)) - Grasshopper Optimization Algorithm (GOA) optimization algorithm.
- **[Grey Wolf Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/grey-wolf-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/grey_wolf_optimizer.py)) - Grey Wolf Optimizer (GWO) optimization algorithm.
- **[Harris Hawks Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/harris-hawks-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/harris_hawks_optimization.py)) - Harris Hawks Optimization (HHO) optimization algorithm.
- **[Honey Badger Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/honey-badger)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/honey_badger.py)) - Honey Badger Algorithm (HBA) optimization algorithm.
- **[Manta Ray Foraging Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/manta-ray)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/manta_ray.py)) - Manta Ray Foraging Optimization (MRFO) optimization algorithm.
- **[Marine Predators Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/marine-predators-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/marine_predators_algorithm.py)) - Marine Predators Algorithm (MPA) optimization algorithm.
- **[Mayfly Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/mayfly-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/mayfly_optimizer.py)) - Mayfly Algorithm (MA) optimization algorithm.
- **[Moth Search Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/moth-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/moth_search.py)) - Moth Search Algorithm (MS) optimization algorithm.
- **[Moth-Flame Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/moth-flame-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/moth_flame_optimization.py)) - Moth-Flame Optimization (MFO) optimization algorithm.
- **[Mountain Gazelle Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/mountain-gazelle)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/mountain_gazelle.py)) - Mountain Gazelle Optimizer (MGO) optimization algorithm.
- **[Orca Predation Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/orca-predator)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/orca_predator.py)) - Orca Predation Algorithm (OPA) optimization algorithm.
- **[Osprey Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/osprey-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/osprey_optimizer.py)) - Osprey Optimization Algorithm (OOA) optimization algorithm.
- **[Particle Swarm Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/particle-swarm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/particle_swarm.py)) - Particle Swarm Optimization (PSO) algorithm.
- **[Pathfinder Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/pathfinder)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/pathfinder.py)) - Pathfinder Algorithm (PFA) optimization algorithm.
- **[Pelican Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/pelican-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/pelican_optimizer.py)) - Pelican Optimization Algorithm (POA) optimization algorithm.
- **[Reptile Search Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/reptile-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/reptile_search.py)) - Reptile Search Algorithm (RSA) optimization algorithm.
- **[Salp Swarm Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/salp-swarm-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/salp_swarm_algorithm.py)) - Salp Swarm Algorithm (SSA) optimization algorithm.
- **[Sand Cat Swarm Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/sand-cat)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/sand_cat.py)) - Sand Cat Swarm Optimization (SCSO) optimization algorithm.
- **[Seagull Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/seagull-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/seagull_optimization.py)) - Seagull Optimization Algorithm (SOA) optimization algorithm.
- **[Slime Mould Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/slime-mould)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/slime_mould.py)) - Slime Mould Algorithm (SMA) optimization algorithm.
- **[Snow Geese Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/snow-geese)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/snow_geese.py)) - Snow Geese Algorithm (SGA) optimization algorithm.
- **[Spotted Hyena Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/spotted-hyena)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/spotted_hyena.py)) - Spotted Hyena Optimizer (SHO) optimization algorithm.
- **[Squirrel Search Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/squirrel-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/squirrel_search.py)) - Squirrel Search Algorithm (SSA) optimization algorithm.
- **[Starling Murmuration Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/starling-murmuration)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/starling_murmuration.py)) - Starling Murmuration Optimizer (SMO) optimization algorithm.
- **[Tunicate Swarm Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/tunicate-swarm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/tunicate_swarm.py)) - Tunicate Swarm Algorithm (TSA) optimization algorithm.
- **[Whale Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/whale-optimization-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/whale_optimization_algorithm.py)) - Whale Optimization Algorithm (WOA) optimization algorithm.
- **[Wild Horse Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/wild-horse)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/wild_horse.py)) - Wild Horse Optimizer (WHO) optimization algorithm.
- **[Zebra Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/swarm-intelligence/zebra-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/swarm_intelligence/zebra_optimizer.py)) - Zebra Optimization Algorithm (ZOA) optimization algorithm.

</details>

<details>
<summary><strong>🧬 Evolutionary (6)</strong></summary>

Algorithms inspired by biological evolution.

- **[Covariance Matrix Adaptation Evolution Strategy](https://anselmoo.github.io/useful-optimizer/algorithms/evolutionary/cma-es)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/cma_es.py)) - Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm.
- **[Cultural Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/evolutionary/cultural-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/cultural_algorithm.py)) - Cultural Algorithm (CA) optimization algorithm.
- **[Differential Evolution](https://anselmoo.github.io/useful-optimizer/algorithms/evolutionary/differential-evolution)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/differential_evolution.py)) - Differential Evolution (DE) optimization algorithm.
- **[Estimation of Distribution Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/evolutionary/estimation-of-distribution-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/estimation_of_distribution_algorithm.py)) - Estimation of Distribution Algorithm (EDA) optimization algorithm.
- **[Genetic Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/evolutionary/genetic-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/genetic_algorithm.py)) - Genetic Algorithm (GA) optimization algorithm.
- **[Imperialist Competitive Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/evolutionary/imperialist-competitive-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/evolutionary/imperialist_competitive_algorithm.py)) - Imperialist Competitive Algorithm (ICA) optimization algorithm.

</details>

<details>
<summary><strong>🧠 Gradient-Based (11)</strong></summary>

Optimization using gradient information.

- **[Adam with Decoupled Weight Decay](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/adamw)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adamw.py)) - Adam with Decoupled Weight Decay (AdamW) optimization algorithm.
- **[Adamax](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/adamax)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adamax.py)) - Adamax optimization algorithm.
- **[Adaptive Delta](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/adadelta)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adadelta.py)) - Adaptive Delta (AdaDelta) optimization algorithm.
- **[Adaptive Gradient Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/adagrad)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adagrad.py)) - Adaptive Gradient Algorithm (AdaGrad) optimization algorithm.
- **[Adaptive Moment Estimation](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/adaptive-moment-estimation)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/adaptive_moment_estimation.py)) - Adaptive Moment Estimation (Adam) optimization algorithm.
- **[AMSGrad](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/amsgrad)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/amsgrad.py)) - AMSGrad optimization algorithm.
- **[Nesterov Accelerated Gradient](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/nesterov-accelerated-gradient)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/nesterov_accelerated_gradient.py)) - Nesterov Accelerated Gradient (NAG) optimization algorithm.
- **[Nesterov-accelerated Adaptive Moment](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/nadam)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/nadam.py)) - Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm.
- **[Root Mean Square Propagation](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/rmsprop)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/rmsprop.py)) - Root Mean Square Propagation (RMSprop) optimization algorithm.
- **[SGD with Momentum](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/sgd-momentum)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/sgd_momentum.py)) - Stochastic Gradient Descent with Momentum (SGD-M) optimization algorithm.
- **[Stochastic Gradient Descent](https://anselmoo.github.io/useful-optimizer/algorithms/gradient-based/stochastic-gradient-descent)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/gradient_based/stochastic_gradient_descent.py)) - Stochastic Gradient Descent (SGD) optimization algorithm.

</details>

<details>
<summary><strong>🎯 Classical (9)</strong></summary>

Traditional mathematical optimization methods.

- **[Broyden-Fletcher-Goldfarb-Shanno](https://anselmoo.github.io/useful-optimizer/algorithms/classical/bfgs)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/bfgs.py)) - Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton optimization algorithm.
- **[Conjugate Gradient Method](https://anselmoo.github.io/useful-optimizer/algorithms/classical/conjugate-gradient)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/conjugate_gradient.py)) - Conjugate Gradient Method (CG) optimization algorithm.
- **[Hill Climbing](https://anselmoo.github.io/useful-optimizer/algorithms/classical/hill-climbing)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/hill_climbing.py)) - Hill Climbing local search optimization algorithm.
- **[Limited-memory BFGS](https://anselmoo.github.io/useful-optimizer/algorithms/classical/lbfgs)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/lbfgs.py)) - Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization algorithm.
- **[Nelder-Mead Simplex Method](https://anselmoo.github.io/useful-optimizer/algorithms/classical/nelder-mead)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/nelder_mead.py)) - Nelder-Mead Simplex Method optimization algorithm.
- **[Powell's Conjugate Direction Method](https://anselmoo.github.io/useful-optimizer/algorithms/classical/powell)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/powell.py)) - Powell's Conjugate Direction Method optimization algorithm.
- **[Simulated Annealing](https://anselmoo.github.io/useful-optimizer/algorithms/classical/simulated-annealing)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/simulated_annealing.py)) - Simulated Annealing (SA) metaheuristic optimization algorithm.
- **[Tabu Search](https://anselmoo.github.io/useful-optimizer/algorithms/classical/tabu-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/tabu_search.py)) - Tabu Search metaheuristic optimization algorithm.
- **[Trust Region Method](https://anselmoo.github.io/useful-optimizer/algorithms/classical/trust-region)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/classical/trust_region.py)) - Trust Region optimization algorithm.

</details>

<details>
<summary><strong>🔧 Metaheuristic (14)</strong></summary>

High-level problem-independent algorithms.

- **[Arithmetic Optimization Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/arithmetic-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/arithmetic_optimization.py)) - Arithmetic Optimization Algorithm (AOA) optimization algorithm.
- **[Colliding Bodies Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/colliding-bodies-optimization)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/colliding_bodies_optimization.py)) - Colliding Bodies Optimization (CBO) optimization algorithm.
- **[Cross-Entropy Method](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/cross-entropy-method)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/cross_entropy_method.py)) - Cross-Entropy Method (CEM) optimization algorithm.
- **[Eagle Strategy](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/eagle-strategy)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/eagle_strategy.py)) - Eagle Strategy (ES) optimization algorithm.
- **[Forensic-Based Investigation Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/forensic-based)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/forensic_based.py)) - Forensic-Based Investigation Optimizer (FBI) optimization algorithm.
- **[Harmony Search](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/harmony-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/harmony_search.py)) - Harmony Search (HS) optimization algorithm.
- **[Sequential Monte Carlo Particle Filter](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/particle-filter)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/particle_filter.py)) - Sequential Monte Carlo Particle Filter (SMC-PF) optimization algorithm.
- **[Shuffled Frog Leaping Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/shuffled-frog-leaping-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/shuffled_frog_leaping_algorithm.py)) - Shuffled Frog Leaping Algorithm (SFLA) optimization algorithm.
- **[Sine Cosine Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/sine-cosine-algorithm)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/sine_cosine_algorithm.py)) - Sine Cosine Algorithm (SCA) optimization algorithm.
- **[Stochastic Diffusion Search](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/stochastic-diffusion-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/stochastic_diffusion_search.py)) - Stochastic Diffusion Search (SDS) optimization algorithm.
- **[Stochastic Fractal Search](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/stochastic-fractal-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/stochastic_fractal_search.py)) - Stochastic Fractal Search (SFS) optimization algorithm.
- **[Variable Depth Search](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/variable-depth-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/variable_depth_search.py)) - Variable Depth Search (VDS) optimization algorithm.
- **[Variable Neighborhood Search](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/variable-neighborhood-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/variable_neighborhood_search.py)) - Variable Neighborhood Search (VNS) optimization algorithm.
- **[Very Large Scale Neighborhood Search](https://anselmoo.github.io/useful-optimizer/algorithms/metaheuristic/very-large-scale-neighborhood-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/metaheuristic/very_large_scale_neighborhood_search.py)) - Very Large Scale Neighborhood Search (VLSN) optimization algorithm.

</details>

<details>
<summary><strong>⚛️ Physics-Inspired (4)</strong></summary>

Algorithms based on physical phenomena.

- **[Atom Search Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/physics-inspired/atom-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/physics_inspired/atom_search.py)) - Atom Search Optimization (ASO) algorithm.
- **[Equilibrium Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/physics-inspired/equilibrium-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/physics_inspired/equilibrium_optimizer.py)) - Equilibrium Optimizer (EO) optimization algorithm.
- **[Gravitational Search Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/physics-inspired/gravitational-search)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/physics_inspired/gravitational_search.py)) - Gravitational Search Algorithm (GSA) optimization algorithm.
- **[RIME: A Physics-based Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/physics-inspired/rime-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/physics_inspired/rime_optimizer.py)) - RIME Optimization Algorithm (RIME) optimization algorithm.

</details>

<details>
<summary><strong>📊 Probabilistic (5)</strong></summary>

Stochastic and Bayesian optimization methods.

- **[Adaptive Metropolis Algorithm](https://anselmoo.github.io/useful-optimizer/algorithms/probabilistic/adaptive-metropolis)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/adaptive_metropolis.py)) - Adaptive Metropolis (AM) algorithm with covariance adaptation.
- **[Bayesian Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/probabilistic/bayesian-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/bayesian_optimizer.py)) - Bayesian Optimization (BO) using Gaussian Process surrogates.
- **[Linear Discriminant Analysis Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/probabilistic/linear-discriminant-analysis)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/linear_discriminant_analysis.py)) - LDA-guided Optimization using Linear Discriminant Analysis.
- **[Sequential Monte Carlo Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/probabilistic/sequential-monte-carlo)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/sequential_monte_carlo.py)) - Sequential Monte Carlo (SMC) optimization with particle filtering.
- **[Tree-structured Parzen Estimator](https://anselmoo.github.io/useful-optimizer/algorithms/probabilistic/parzen-tree-estimator)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/probabilistic/parzen_tree_estimator.py)) - Tree-structured Parzen Estimator (TPE) for hyperparameter optimization.

</details>

<details>
<summary><strong>👥 Social-Inspired (4)</strong></summary>

Algorithms inspired by social behaviors.

- **[Political Optimizer](https://anselmoo.github.io/useful-optimizer/algorithms/social-inspired/political-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/social_inspired/political_optimizer.py)) - Political Optimizer (PO) algorithm.
- **[Soccer League Competition](https://anselmoo.github.io/useful-optimizer/algorithms/social-inspired/soccer-league-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/social_inspired/soccer_league_optimizer.py)) - Soccer League Competition (SLC) algorithm.
- **[Social Group Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/social-inspired/social-group-optimizer)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/social_inspired/social_group_optimizer.py)) - Social Group Optimization (SGO) algorithm.
- **[Teaching-Learning Based Optimization](https://anselmoo.github.io/useful-optimizer/algorithms/social-inspired/teaching-learning)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/social_inspired/teaching_learning.py)) - Teaching-Learning Based Optimization (TLBO) algorithm.

</details>

<details>
<summary><strong>🔒 Constrained (5)</strong></summary>

Optimization with constraints.

- **[Augmented Lagrangian Method](https://anselmoo.github.io/useful-optimizer/algorithms/constrained/augmented-lagrangian-method)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/constrained/augmented_lagrangian_method.py)) - Augmented Lagrangian Method (ALM) optimization algorithm.
- **[Barrier Method (Interior Point)](https://anselmoo.github.io/useful-optimizer/algorithms/constrained/barrier-method)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/constrained/barrier_method.py)) - Barrier Method (Interior Point Method) optimization algorithm.
- **[Penalty Method (Quadratic Penalty)](https://anselmoo.github.io/useful-optimizer/algorithms/constrained/penalty-method)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/constrained/penalty_method.py)) - Penalty Method for constrained optimization.
- **[Sequential Quadratic Programming](https://anselmoo.github.io/useful-optimizer/algorithms/constrained/sequential-quadratic-programming)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/constrained/sequential_quadratic_programming.py)) - Sequential Quadratic Programming (SQP) optimization algorithm.
- **[Successive Linear Programming](https://anselmoo.github.io/useful-optimizer/algorithms/constrained/successive-linear-programming)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/constrained/successive_linear_programming.py)) - Successive Linear Programming (SLP) optimization algorithm.

</details>

<details>
<summary><strong>📈 Multi-Objective (3)</strong></summary>

Pareto-optimal solutions for multiple objectives.

- **[Multi-Objective Evolutionary Algorithm based on Decomposition](https://anselmoo.github.io/useful-optimizer/algorithms/multi-objective/moead)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/multi_objective/moead.py)) - Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D).
- **[Non-dominated Sorting Genetic Algorithm II](https://anselmoo.github.io/useful-optimizer/algorithms/multi-objective/nsga-ii)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/multi_objective/nsga_ii.py)) - Non-dominated Sorting Genetic Algorithm II (NSGA-II) multi-objective optimizer.
- **[Strength Pareto Evolutionary Algorithm 2](https://anselmoo.github.io/useful-optimizer/algorithms/multi-objective/spea2)** ([source](https://github.com/Anselmoo/useful-optimizer/blob/main/src/opt/multi_objective/spea2.py)) - Strength Pareto Evolutionary Algorithm 2 (SPEA2) multi-objective optimizer.

</details>

<!-- END GENERATED: readme-optimizers -->

> [!NOTE]
> Please note that not all of these algorithms are suitable for all types of optimization problems. Some are better suited for continuous optimization problems, some for discrete optimization problems, and others for specific types of problems like quadratic programming or linear discriminant analysis.

## Testing and Documentation

Useful Optimizer includes comprehensive doctests to ensure all examples in the documentation are correct and up-to-date. All optimizer classes and benchmark functions include working examples that can be verified automatically.

### Documentation Generation

The project uses an automated documentation generator with Griffe integration to create comprehensive API documentation:

```bash
# Generate all documentation (markdown + JSON + API)
uv run python scripts/generate_docs.py --all --json --full-api --sidebar

# Or use the npm shortcut
cd docs && npm run docs:api:all
```

This generates:
- **Markdown pages** for every optimizer in `docs/algorithms/`, plus category indexes and API pages
- **JSON metadata** at `docs/public/optimizers/optimizers.json`
- **VitePress sidebar** auto-configuration
- **Griffe API dump** `docs/api/full_api.json` for the entire opt package (generated in CI, not committed)

Performance: ~6 seconds for complete generation

See [`docs/DOCUMENTATION_GENERATOR.md`](docs/DOCUMENTATION_GENERATOR.md) for detailed usage.

### Running Tests

To run all tests including doctests:

```bash
# Using pytest with doctests
uv run pytest --doctest-modules --doctest-glob="*.md" -v

# Run only doctests for a specific module
uv run pytest opt/benchmark/functions.py --doctest-modules -v

# Run only doctests for optimizers
uv run pytest opt/swarm_intelligence/particle_swarm.py --doctest-modules -v
```

### Example Doctest Usage

All benchmark functions include working examples in their docstrings:

```python
import numpy as np
from opt.benchmark.functions import sphere

# At optimum
result = sphere(np.array([0.0, 0.0]))
print(float(result))  # 0.0

# Away from optimum
result = sphere(np.array([1.0, 1.0]))
print(float(result))  # 2.0
```

All optimizer classes include usage examples in their docstrings:

```python
from opt.swarm_intelligence.particle_swarm import ParticleSwarm
from opt.benchmark.functions import sphere

optimizer = ParticleSwarm(
    func=sphere, dim=2, lower_bound=-5, upper_bound=5,
    max_iter=10, seed=42
)
solution, fitness = optimizer.search()
print(f"Fitness: {fitness}")  # Should be < 1.0
```

These examples serve as both documentation and automated tests, ensuring the code examples in docstrings always work correctly.

## Contributing

Contributions to Useful Optimizer are welcome! Please read the contributing guidelines before getting started.

## [License](LICENSE)

---

> [!WARNING]
> This project was generated with GitHub Copilot and may not be completely verified. Please use with caution and feel free to report any issues you encounter. Thank you!

> [!WARNING]
> Some parts still contain the _legacy_ `np.random.rand` call. See also: https://docs.astral.sh/ruff/rules/numpy-legacy-random/
