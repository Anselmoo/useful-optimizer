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

## Current Implemented Optimizer

The current version of Useful Optimizer includes a wide range of optimization algorithms, each implemented as a separate module. Here's a brief overview of the implemented optimizers:

Sure, here's a brief description of each optimizer:

- **Adadelta, Adagrad, and Adaptive Moment Estimation**: These are gradient-based optimization algorithms commonly used in machine learning and deep learning.

- **AdaMax**: This is an Adam variant that uses the infinity norm for the second moment estimation, making it more robust to large gradients.

- **AdamW**: This is an Adam variant with decoupled weight decay that provides better regularization and improved generalization in machine learning.

- **AMSGrad**: This is an Adam variant with non-decreasing second moment estimates that addresses convergence issues in original Adam.

- **Ant Colony Optimization**: This is a nature-inspired algorithm that mimics the behavior of ants to solve optimization problems.

- **Artificial Fish Swarm Algorithm**: This algorithm simulates the behavior of fish in nature to perform global optimization.

- **Augmented Lagrangian Method**: This is a method to solve constrained optimization problems. It combines the objective function and the constraints into a single function using Lagrange multipliers.

- **Bat Algorithm**: This is a metaheuristic optimization algorithm inspired by the echolocation behavior of microbats.

- **Bee Algorithm**: This is a population-based search algorithm inspired by the food foraging behavior of honey bee colonies.

- **BFGS (Broyden-Fletcher-Goldfarb-Shanno)**: This is a quasi-Newton method that approximates the inverse Hessian matrix for efficient second-order optimization.

- **Cat Swarm Optimization**: This algorithm is based on the behavior of cats and distinguishes between two forms of behavior in cats: seeking mode and tracing mode.

- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: This is an evolutionary algorithm for difficult non-linear non-convex optimization problems in continuous domain.

- **Colliding Bodies Optimization**: This is a physics-inspired optimization method, based on the collision and explosion of bodies.

- **Conjugate Gradient**: This is an efficient iterative method for solving systems of linear equations and optimization problems, particularly effective for quadratic functions.

- **Cross Entropy Method**: This is a Monte Carlo method for importance sampling and optimization.

- **Cuckoo Search**: This is a nature-inspired metaheuristic optimization algorithm, which is based on the obligate brood parasitism of some cuckoo species.

- **Cultural Algorithm**: This is a type of evolutionary algorithm that is based on the concept of culture, or shared information and knowledge.

- **Differential Evolution**: This is a population-based metaheuristic algorithm, which uses mechanisms inspired by biological evolution, such as reproduction, mutation, recombination, and selection.

- **Eagle Strategy**: This is a metaheuristic optimization algorithm inspired by the hunting behavior of eagles.

- **Estimation of Distribution Algorithm**: This is a stochastic optimization algorithm that uses a probabilistic model of candidate solutions.

- **Firefly Algorithm**: This is a nature-inspired metaheuristic optimization algorithm, based on the flashing behavior of fireflies.

- **Genetic Algorithm**: This is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution.

- **Glowworm Swarm Optimization**: This is a nature-inspired optimization algorithm based on the behavior of glowworms.

- **Grey Wolf Optimizer**: This is a metaheuristic optimization algorithm inspired by grey wolves.

- **Harmony Search**: This is a music-inspired metaheuristic optimization algorithm.

- **Hessian Free Optimization**: This is a second-order optimization algorithm that uses information from the Hessian matrix to guide the search.

- **Hill Climbing**: This is a mathematical optimization technique which belongs to the family of local search.

- **Imperialist Competitive Algorithm**: This is a socio-politically motivated global search strategy, which is based on the imperialistic competition.

- **L-BFGS (Limited-memory BFGS)**: This is a limited-memory version of BFGS that is suitable for large-scale optimization problems where storing the full Hessian approximation is impractical.

- **Linear Discriminant Analysis**: This is a method used in statistics, pattern recognition, and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events.

- **Nadam**: This is Nesterov-accelerated Adam that combines the benefits of Adam with Nesterov momentum for improved convergence.

- **Nelder-Mead**: This is a derivative-free simplex-based optimization method that is particularly useful for non-differentiable and noisy functions.

- **Nesterov Accelerated Gradient**: This is an accelerated gradient method that uses lookahead momentum to achieve better convergence rates than standard gradient descent.

- **Particle Filter**: This is a statistical filter technique used to estimate the state of a system where the state model and the measurements are both nonlinear.

- **Particle Swarm Optimization**: This is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality, mimicking the social behavior of bird flocking or fish schooling.

- **Parzen Tree Estimator**: This is a non-parametric method to estimate the density function of random variables.

- **Powell's Method**: This is a derivative-free optimization algorithm that uses conjugate directions to minimize functions without requiring gradient information.

- **RMSprop**: This is an adaptive learning rate optimization algorithm that uses a moving average of squared gradients to normalize the gradient.

- **Shuffled Frog Leaping Algorithm**: This is a metaheuristic optimization algorithm inspired by the memetic evolution of a group of frogs when searching for food.

- **Simulated Annealing**: This is a probabilistic technique for approximating the global optimum of a given function, mimicking the process of heating a material and then slowly lowering the temperature to decrease defects, thus minimizing the system energy.

- **Sine Cosine Algorithm**: This is a new population-based meta-heuristic algorithm, inspired by the mathematical sine and cosine functions.

- **Squirrel Search Algorithm**: This is a new nature-inspired metaheuristic optimization algorithm, inspired by the caching behavior of squirrels.

- **Stochastic Diffusion Search**: This is a population-based search algorithm, based on the behavior of ants when searching for food.

- **Stochastic Fractal Search**: This is a metaheuristic search algorithm inspired by the natural phenomenon of fractal shapes and Brownian motion.

- **Stochastic Gradient Descent (SGD)**: This is a fundamental gradient-based optimization algorithm that updates parameters in the direction opposite to the gradient of the objective function.

- **SGD with Momentum**: This is SGD enhanced with momentum that accelerates convergence and helps navigate through local minima by accumulating velocity in consistent gradient directions.

- **Successive Linear Programming**: This is an optimization method for nonlinear optimization problems.

- **Tabu Search**: This is a metaheuristic search method employing local search methods used for mathematical optimization.

- **Trust Region**: This is a robust optimization method that iteratively solves optimization problems within a region where a model function is trusted to be an adequate representation.

- **Variable Depth Search**: This is a search algorithm that explores the search space by variable-depth first search and backtracking.

- **Variable Neighbourhood Search**: This is a metaheuristic search method for discrete optimization problems.

- **Very Large Scale Neighborhood Search**: This is a search method that explores very large neighborhoods with an efficient algorithm.

- **Whale Optimization Algorithm**: This is a new bio-inspired optimization algorithm, which simulates the social behavior of humpback whales.

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
