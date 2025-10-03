[![DOI](https://zenodo.org/badge/776526436.svg)](https://zenodo.org/doi/10.5281/zenodo.13294276)

# Useful Optimizer

Useful Optimizer is a dedicated set of optimization algorithms for numeric problems. It's designed to provide a comprehensive collection of optimization techniques that can be easily used and integrated into any project.

## Version

The current version of Useful Optimizer is 0.1.0.

## Features

- A wide range of optimization algorithms.
- Easy to use and integrate.
- Suitable for various numeric problems.
- Having fun to play with the algorithms

## Installation

To install Useful Optimizer, you can use pip:

```bash
pip install git+https://github.com/Anselmoo/useful-optimizer
```

## Usage

Here's a basic example of how to use Useful Optimizer:

```python
from opt.benchmark import CrossEntropyMethod
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

You can also use the new gradient-based optimizers:

```python
from opt.stochastic_gradient_descent import SGD
from opt.adamw import AdamW
from opt.bfgs import BFGS

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

## Implemented Optimizers

The current version of Useful Optimizer includes 54 optimization algorithms, each implemented as a separate module. Each optimizer is linked to its corresponding source code for easy reference and study.

<details>
<summary><strong>🧠 Gradient-Based Optimizers</strong></summary>

These optimizers use gradient information to guide the search process and are commonly used in machine learning and deep learning applications.

- **[Adadelta](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/adadelta.py)** - An adaptive learning rate method that uses only first-order information
- **[Adagrad](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/adagrad.py)** - Adapts the learning rate to the parameters, performing smaller updates for frequently occurring features
- **[Adaptive Moment Estimation (Adam)](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/adaptive_moment_estimation.py)** - Combines advantages of AdaGrad and RMSProp with bias correction
- **[AdaMax](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/adamax.py)** - Adam variant using infinity norm for second moment estimation
- **[AdamW](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/adamw.py)** - Adam with decoupled weight decay for better regularization
- **[AMSGrad](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/amsgrad.py)** - Adam variant with non-decreasing second moment estimates
- **[BFGS](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/bfgs.py)** - Quasi-Newton method approximating the inverse Hessian matrix
- **[Conjugate Gradient](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/conjugate_gradient.py)** - Efficient iterative method for solving systems of linear equations
- **[L-BFGS](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/lbfgs.py)** - Limited-memory version of BFGS for large-scale optimization
- **[Nadam](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/nadam.py)** - Nesterov-accelerated Adam combining Adam with Nesterov momentum
- **[Nesterov Accelerated Gradient](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/nesterov_accelerated_gradient.py)** - Accelerated gradient method with lookahead momentum
- **[RMSprop](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/rmsprop.py)** - Adaptive learning rate using moving average of squared gradients
- **[SGD with Momentum](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/sgd_momentum.py)** - SGD enhanced with momentum for faster convergence
- **[Stochastic Gradient Descent](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/stochastic_gradient_descent.py)** - Fundamental gradient-based optimization algorithm

</details>

<details>
<summary><strong>🦋 Nature-Inspired Metaheuristics</strong></summary>

These algorithms are inspired by natural phenomena and biological behaviors to solve optimization problems.

- **[Ant Colony Optimization](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/ant_colony.py)** - Mimics ant behavior for finding optimal paths
- **[Artificial Fish Swarm Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/artificial_fish_swarm_algorithm.py)** - Simulates fish behavior for global optimization
- **[Bat Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/bat_algorithm.py)** - Inspired by echolocation behavior of microbats
- **[Bee Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/bee_algorithm.py)** - Based on honey bee food foraging behavior
- **[Cat Swarm Optimization](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/cat_swarm_optimization.py)** - Models cat behavior with seeking and tracing modes
- **[Cuckoo Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/cuckoo_search.py)** - Based on obligate brood parasitism of cuckoo species
- **[Eagle Strategy](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/eagle_strategy.py)** - Inspired by hunting behavior of eagles
- **[Firefly Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/firefly_algorithm.py)** - Based on flashing behavior of fireflies
- **[Glowworm Swarm Optimization](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/glowworm_swarm_optimization.py)** - Inspired by glowworm behavior
- **[Grey Wolf Optimizer](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/grey_wolf_optimizer.py)** - Mimics leadership hierarchy and hunting of grey wolves
- **[Particle Swarm Optimization](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/particle_swarm.py)** - Simulates social behavior of bird flocking or fish schooling
- **[Shuffled Frog Leaping Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/shuffled_frog_leaping_algorithm.py)** - Inspired by memetic evolution of frogs searching for food
- **[Squirrel Search Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/squirrel_search.py)** - Based on caching behavior of squirrels
- **[Whale Optimization Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/whale_optimization_algorithm.py)** - Simulates social behavior of humpback whales

</details>

<details>
<summary><strong>🧬 Evolutionary and Population-Based Algorithms</strong></summary>

These algorithms use principles of evolution and population dynamics to find optimal solutions.

- **[CMA-ES](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/cma_es.py)** - Covariance Matrix Adaptation Evolution Strategy for continuous optimization
- **[Cultural Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/cultural_algorithm.py)** - Evolutionary algorithm based on cultural evolution
- **[Differential Evolution](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/differential_evolution.py)** - Population-based algorithm using biological evolution mechanisms
- **[Estimation of Distribution Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/estimation_of_distribution_algorithm.py)** - Uses probabilistic model of candidate solutions
- **[Genetic Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/genetic_algorithm.py)** - Inspired by Charles Darwin's theory of natural evolution
- **[Imperialist Competitive Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/imperialist_competitive_algorithm.py)** - Based on imperialistic competition

</details>

<details>
<summary><strong>🎯 Local Search and Classical Methods</strong></summary>

Traditional optimization methods include local search techniques and classical mathematical approaches.

- **[Hill Climbing](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/hill_climbing.py)** - Local search algorithm that continuously moves toward increasing value
- **[Nelder-Mead](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/nelder_mead.py)** - Derivative-free simplex method for optimization
- **[Powell's Method](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/powell.py)** - Derivative-free optimization using conjugate directions
- **[Simulated Annealing](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/simulated_annealing.py)** - Probabilistic technique mimicking the annealing process in metallurgy
- **[Tabu Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/tabu_search.py)** - Metaheuristic using memory structures to avoid cycles
- **[Variable Depth Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/variable_depth_search.py)** - Explores search space with variable-depth first search
- **[Variable Neighbourhood Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/variable_neighbourhood_search.py)** - Metaheuristic for discrete optimization problems
- **[Very Large Scale Neighborhood Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/very_large_scale_neighborhood_search.py)** - Explores very large neighborhoods efficiently

</details>

<details>
<summary><strong>🔬 Physics and Mathematical-Inspired Algorithms</strong></summary>

Algorithms inspired by physical phenomena and mathematical concepts.

- **[Colliding Bodies Optimization](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/colliding_bodies_optimization.py)** - Physics-inspired method based on collision and explosion
- **[Harmony Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/harmony_search.py)** - Music-inspired metaheuristic optimization
- **[Sine Cosine Algorithm](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/sine_cosine_algorithm.py)** - Based on mathematical sine and cosine functions
- **[Stochastic Diffusion Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/stochastic_diffusion_search.py)** - Population-based search inspired by diffusion processes
- **[Stochastic Fractal Search](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/stochastic_fractal_search.py)** - Inspired by fractal shapes and Brownian motion

</details>

<details>
<summary><strong>📊 Statistical and Probabilistic Methods</strong></summary>

Methods based on statistical inference and probabilistic approaches.

- **[Cross Entropy Method](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/cross_entropy_method.py)** - Monte Carlo method for importance sampling and optimization
- **[Particle Filter](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/particle_filter.py)** - Statistical filter for nonlinear state estimation
- **[Parzen Tree Estimator](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/parzen_tree_stimator.py)** - Non-parametric density estimation method

</details>

<details>
<summary><strong>🔧 Specialized and Constrained Optimization</strong></summary>

Specialized algorithms for particular types of optimization problems.

- **[Augmented Lagrangian Method](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/augmented_lagrangian_method.py)** - Method for solving constrained optimization problems
- **[Linear Discriminant Analysis](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/linear_discriminant_analysis.py)** - Statistical method for dimensionality reduction and classification
- **[Successive Linear Programming](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/successive_linear_programming.py)** - Method for nonlinear optimization using linear approximations
- **[Trust Region](https://github.com/Anselmoo/useful-optimizer/blob/main/opt/trust_region.py)** - Robust optimization method using trusted model regions

</details>

> [!NOTE]
> Please note that not all of these algorithms are suitable for all types of optimization problems. Some are better suited for continuous optimization problems, some for discrete optimization problems, and others for specific types of problems like quadratic programming or linear discriminant analysis.

## Contributing

Contributions to Useful Optimizer are welcome! Please read the contributing guidelines before getting started.

## [License](LICENSE)

---

> [!WARNING]
> This project was generated with GitHub Copilot and may not be completely verified. Please use with caution and feel free to report any issues you encounter. Thank you!

> [!WARNING]
> Some parts still contain the _legacy_ `np.random.rand` call. See also: https://docs.astral.sh/ruff/rules/numpy-legacy-random/
