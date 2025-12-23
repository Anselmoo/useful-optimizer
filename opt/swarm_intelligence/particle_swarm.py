"""Particle Swarm Optimization (PSO) algorithm implementation.

This module provides an implementation of the Particle Swarm Optimization (PSO) algorithm for solving optimization problems.
PSO is a population-based stochastic optimization algorithm inspired by the social behavior of bird flocking or fish schooling.

The main class in this module is `ParticleSwarm`, which represents the PSO algorithm. It takes an objective function, lower and upper bounds of the search space, dimensionality of the search space, and other optional parameters as input. The `search` method performs the PSO optimization and returns the best solution found.

Example usage:
    optimizer = ParticleSwarm(
        func=shifted_ackley,
        lower_bound=-32.768,
        upper_bound=+32.768,
        dim=2,
        population_size=100,
        max_iter=1000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness value: {best_fitness}")

Classes:
    - ParticleSwarm: Particle Swarm Optimization (PSO) algorithm for optimization problems.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.constants import DEFAULT_MAX_ITERATIONS
from opt.constants import DEFAULT_POPULATION_SIZE
from opt.constants import PSO_COGNITIVE_COEFFICIENT
from opt.constants import PSO_INERTIA_WEIGHT
from opt.constants import PSO_SOCIAL_COEFFICIENT


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ParticleSwarm(AbstractOptimizer):
    r"""FIXME: [Algorithm Full Name] ([ACRONYM]) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | FIXME: [Full algorithm name]             |
        | Acronym           | FIXME: [SHORT]                           |
        | Year Introduced   | FIXME: [YYYY]                            |
        | Authors           | FIXME: [Last, First; ...]                |
        | Algorithm Class   | Swarm Intelligence |
        | Complexity        | FIXME: O([expression])                   |
        | Properties        | FIXME: [Population-based, ...]           |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        FIXME: Core update equation:

            $$
            x_{t+1} = x_t + v_t
            $$

        where:
            - $x_t$ is the position at iteration $t$
            - $v_t$ is the velocity/step at iteration $t$
            - FIXME: Additional variable definitions...

        Constraint handling:
            - **Boundary conditions**: FIXME: [clamping/reflection/periodic]
            - **Feasibility enforcement**: FIXME: [description]

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | population_size        | 100     | 10*dim           | Number of individuals          |
        | max_iter               | 1000    | 10000            | Maximum iterations             |
        | FIXME: [param_name]    | [val]   | [bbob_val]       | [description]                  |

        **Sensitivity Analysis**:
            - FIXME: `[param_name]`: **[High/Medium/Low]** impact on convergence
            - Recommended tuning ranges: FIXME: $\text{[param]} \in [\text{min}, \text{max}]$

    COCO/BBOB Benchmark Settings:
        **Search Space**:
            - Dimensions tested: `2, 3, 5, 10, 20, 40`
            - Bounds: Function-specific (typically `[-5, 5]` or `[-100, 100]`)
            - Instances: **15** per function (BBOB standard)

        **Evaluation Budget**:
            - Budget: $\text{dim} \times 10000$ function evaluations
            - Independent runs: **15** (for statistical significance)
            - Seeds: `0-14` (reproducibility requirement)

        **Performance Metrics**:
            - Target precision: `1e-8` (BBOB default)
            - Success rate at precision thresholds: `[1e-8, 1e-6, 1e-4, 1e-2]`
            - Expected Running Time (ERT) tracking

    Example:
        Basic usage with BBOB benchmark function:

        >>> from opt.swarm_intelligence.particle_swarm import ParticleSwarm
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ParticleSwarm(
        ...     func=shifted_ackley,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     dim=2,
        ...     max_iter=100,
        ...     seed=42,  # Required for reproducibility
        ... )
        >>> solution, fitness = optimizer.search()
        >>> isinstance(fitness, float) and fitness >= 0
        True

        COCO benchmark example:

        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ParticleSwarm(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, lower_bound, upper_bound, dim, population_size, max_iter, c1, c2, w, seed, track_history

        Common parameters (adjust based on actual signature):
        func (Callable[[ndarray], float]): Objective function to minimize. Must accept
            numpy array and return scalar. BBOB functions available in
            `opt.benchmark.functions`.
        lower_bound (float): Lower bound of search space. BBOB typical: -5
            (most functions).
        upper_bound (float): Upper bound of search space. BBOB typical: 5
            (most functions).
        dim (int): Problem dimensionality. BBOB standard dimensions: 2, 3, 5, 10, 20, 40.
        max_iter (int, optional): Maximum iterations. BBOB recommendation: 10000 for
            complete evaluation. Defaults to 1000.
        seed (int | None, optional): Random seed for reproducibility. BBOB requires
            seeds 0-14 for 15 runs. If None, generates random seed. Defaults to None.
        population_size (int, optional): Population size. BBOB recommendation: 10*dim
            for population-based methods. Defaults to 100. (Only for population-based
            algorithms)
        track_history (bool, optional): Enable convergence history tracking for BBOB
            post-processing. Defaults to False.
        FIXME: [algorithm_specific_params] ([type], optional): FIXME: Document any
            algorithm-specific parameters not listed above. Defaults to [value].

    Attributes:
        func (Callable[[ndarray], float]): The objective function being optimized.
        lower_bound (float): Lower search space boundary.
        upper_bound (float): Upper search space boundary.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum number of iterations.
        seed (int): **REQUIRED** Random seed for reproducibility (BBOB compliance).
        population_size (int): Number of individuals in population.
        track_history (bool): Whether convergence history is tracked.
        history (dict[str, list]): Optimization history if track_history=True. Contains:
            - 'best_fitness': list[float] - Best fitness per iteration
            - 'best_solution': list[ndarray] - Best solution per iteration
            - 'population_fitness': list[ndarray] - All fitness values
            - 'population': list[ndarray] - All solutions
        FIXME: [algorithm_specific_attrs] ([type]): FIXME: [Description]

    Methods:
        search() -> tuple[np.ndarray, float]:
            Execute optimization algorithm.

    Returns:
        tuple[np.ndarray, float]:
        Best solution found and its fitness value

    Raises:
        ValueError: If search space is invalid or function evaluation fails.

    Notes:
        - Modifies self.history if track_history=True
        - Uses self.seed for all random number generation
        - BBOB: Returns final best solution after max_iter or convergence

    References:
        FIXME: [1] Author1, A., Author2, B. (YEAR). "Algorithm Name: Description."
        _Journal Name_, Volume(Issue), Pages.
        https://doi.org/10.xxxx/xxxxx

        [2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
            "COCO: A platform for comparing continuous optimizers in a black-box setting."
            _Optimization Methods and Software_, 36(1), 114-144.
            https://doi.org/10.1080/10556788.2020.1808977

        **COCO Data Archive**:
            - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
            - FIXME: Algorithm data: [URL to algorithm-specific COCO results if available]
            - Code repository: https://github.com/Anselmoo/useful-optimizer

        **Implementation**:
            - FIXME: Original paper code: [URL if different from this implementation]
            - This implementation: Based on [1] with modifications for BBOB compliance

    See Also:
        FIXME: [RelatedAlgorithm1]: Similar algorithm with [key difference]
            BBOB Comparison: [Brief performance notes on sphere/rosenbrock/ackley]

        FIXME: [RelatedAlgorithm2]: [Relationship description]
            BBOB Comparison: Generally [faster/slower/more robust] on [function classes]

        AbstractOptimizer: Base class for all optimizers
        opt.benchmark.functions: BBOB-compatible test functions

        Related BBOB Algorithm Classes:
            - Evolutionary: GeneticAlgorithm, DifferentialEvolution
            - Swarm: ParticleSwarm, AntColony
            - Gradient: AdamW, SGDMomentum

    Notes:
        **Computational Complexity**:
        - Time per iteration: FIXME: $O(\text{[expression]})$
        - Space complexity: FIXME: $O(\text{[expression]})$
        - BBOB budget usage: FIXME: _[Typical percentage of dim*10000 budget needed]_

        **BBOB Performance Characteristics**:
            - **Best function classes**: FIXME: [Unimodal/Multimodal/Ill-conditioned/...]
            - **Weak function classes**: FIXME: [Function types where algorithm struggles]
            - Typical success rate at 1e-8 precision: FIXME: **[X]%** (dim=5)
            - Expected Running Time (ERT): FIXME: [Comparative notes vs other algorithms]

        **Convergence Properties**:
            - Convergence rate: FIXME: [Linear/Quadratic/Exponential]
            - Local vs Global: FIXME: [Tendency for local/global optima]
            - Premature convergence risk: FIXME: **[High/Medium/Low]**

        **Reproducibility**:
            - **Deterministic**: FIXME: [Yes/No] - Same seed guarantees same results
            - **BBOB compliance**: seed parameter required for 15 independent runs
            - Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
            - RNG usage: `numpy.random.default_rng(self.seed)` throughout

        **Implementation Details**:
            - Parallelization: FIXME: [Not supported/Supported via `[method]`]
            - Constraint handling: FIXME: [Clamping to bounds/Penalty/Repair]
            - Numerical stability: FIXME: [Considerations for floating-point arithmetic]

        **Known Limitations**:
            - FIXME: [Any known issues or limitations specific to this implementation]
            - FIXME: BBOB known issues: [Any BBOB-specific challenges]

        **Version History**:
            - v0.1.0: Initial implementation
            - FIXME: [vX.X.X]: [Changes relevant to BBOB compliance]
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = DEFAULT_POPULATION_SIZE,
        max_iter: int = DEFAULT_MAX_ITERATIONS,
        c1: float = PSO_COGNITIVE_COEFFICIENT,
        c2: float = PSO_SOCIAL_COEFFICIENT,
        w: float = PSO_INERTIA_WEIGHT,
        seed: int | None = None,
        track_history: bool = False,
    ) -> None:
        """Initialize the ParticleSwarm class.

        Args:
            func (Callable[[ndarray], float]): The objective function to be minimized.
            lower_bound (float): The lower bound of the search space.
            upper_bound (float): The upper bound of the search space.
            dim (int): The dimensionality of the search space.
            population_size (int, optional): The number of particles in the swarm (default: 100).
            max_iter (int, optional): The maximum number of iterations (default: 1000).
            c1 (float, optional): The cognitive parameter (default: 1.5).
            c2 (float, optional): The social parameter (default: 1.5).
            w (float, optional): The inertia weight (default: 0.5).
            seed (int | None, optional): The seed for the random number generator (default: None).
            track_history (bool, optional): Whether to track optimization history for visualization (default: False).
        """
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
            track_history=track_history,
        )
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the particle swarm optimization.

        Returns:
        tuple[np.ndarray, float]: A tuple containing the best position found and its corresponding fitness value.
        """
        # Initialize population and fitness
        population = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(self.func, 1, population)

        # Initialize velocity
        velocity = np.zeros((self.population_size, self.dim))

        # Initialize best position and fitness
        best_position = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        # Main loop
        for _ in range(self.max_iter):
            # Track history if enabled
            if self.track_history:
                self.history["best_fitness"].append(float(best_fitness))
                self.history["best_solution"].append(best_position.copy())
                self.history["population_fitness"].append(fitness.copy())
                self.history["population"].append(population.copy())

            self.seed += 1
            # Update velocity
            r1 = np.random.default_rng(self.seed + 1).random(
                (self.population_size, self.dim)
            )
            r2 = np.random.default_rng(self.seed + 2).random(
                (self.population_size, self.dim)
            )
            velocity = (
                self.w * velocity
                + self.c1 * r1 * (best_position - population)
                + self.c2 * r2 * (population[np.argmin(fitness)] - population)
            )

            # Update position
            population += velocity

            # Ensure the position stays within the bounds
            population = np.clip(population, self.lower_bound, self.upper_bound)

            # Update fitness
            fitness = np.apply_along_axis(self.func, 1, population)

            # Update best position and fitness
            best_index = np.argmin(fitness)
            if fitness[best_index] < best_fitness:
                best_position = population[best_index]
                best_fitness = fitness[best_index]

        # Track final state
        if self.track_history:
            self.history["best_fitness"].append(float(best_fitness))
            self.history["best_solution"].append(best_position.copy())
            self.history["population_fitness"].append(fitness.copy())
            self.history["population"].append(population.copy())

        return best_position, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ParticleSwarm)
