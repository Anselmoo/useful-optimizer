"""Glowworm Swarm Optimization (GSO) algorithm.

This module implements the Glowworm Swarm Optimization (GSO) algorithm as an optimizer.
GSO is a population-based optimization algorithm inspired by the behavior of glowworms.
It is commonly used to solve optimization problems.

The GlowwormSwarmOptimization class provides an implementation of the GSO algorithm. It
takes an objective function, lower and upper bounds of the search space, dimensionality
of the search space, and other optional parameters as input. The algorithm searches for
the best solution within the given search space by iteratively updating the positions of
glowworms based on their luciferin levels and neighboring glowworms.

Usage:
    optimizer = GlowwormSwarmOptimization(
        func=shifted_ackley, dim=2, lower_bound=-32.768, upper_bound=+32.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    luciferin_decay (float): The decay rate of luciferin.
    randomness (float): The randomness factor for glowworm movement.
    step_size (float): The step size for glowworm movement.

Methods:
    _initialize(): Initialize the population of glowworms.
    _compute_fitness(population): Compute the fitness values for the glowworm population.
    _update_luciferin(population, fitness): Update the luciferin levels of the glowworms.
    _move_glowworms(population, luciferin): Move the glowworms based on their luciferin levels.
    search(): Run the glowworm swarm optimization algorithm and return the best solution and fitness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class GlowwormSwarmOptimization(AbstractOptimizer):
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

        >>> from opt.swarm_intelligence.glowworm_swarm_optimization import GlowwormSwarmOptimization
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = GlowwormSwarmOptimization(
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
        >>> optimizer = GlowwormSwarmOptimization(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> len(solution) == 10
        True

    Args:
        FIXME: Document all parameters with BBOB guidance.
        Detected parameters from __init__ signature: func, lower_bound, upper_bound, dim, population_size, max_iter, luciferin_decay, randomness, step_size, seed

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
        population_size: int = 100,
        max_iter: int = 1000,
        luciferin_decay: float = 0.1,
        randomness: float = 0.5,
        step_size: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the GlowwormSwarmOptimization class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=population_size,
        )
        self.luciferin_decay = luciferin_decay
        self.randomness = randomness
        self.step_size = step_size

    def _initialize(self) -> ndarray:
        """Initializes the population of glowworms with random positions.

        Returns:
        ndarray: The initialized population of glowworms.
        """
        return np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def _compute_fitness(self, population: ndarray) -> ndarray:
        """Compute the fitness values for the given population.

        Args:
            population (ndarray): Population for which to compute fitness values.

        Returns:
        ndarray: Computed fitness values for the population.
        """
        return np.apply_along_axis(self.func, 1, population)

    def _update_luciferin(self, population: ndarray, fitness: ndarray) -> ndarray:
        """Update the luciferin levels of the glowworms based on their fitness values.

        Args:
            population (ndarray): The population of glowworms.
            fitness (ndarray): The fitness values of the glowworms.

        Returns:
        ndarray: The updated luciferin levels of the glowworms.
        """
        return (1 - self.luciferin_decay) * fitness / np.linalg.norm(population, axis=1)

    def _move_glowworms(self, population: ndarray, luciferin: ndarray) -> ndarray:
        """Moves the glowworms in the population based on their luciferin levels and neighboring glowworms.

        Args:
            population (ndarray): The current population of glowworms.
            luciferin (ndarray): The luciferin levels of the glowworms.

        Returns:
        ndarray: The new population of glowworms after moving.

        """
        new_population = []
        for i in range(self.population_size):
            self.seed += 1
            glowworm = population[i]
            neighbors = population[luciferin > luciferin[i]]
            if len(neighbors) > 0:
                distances = np.linalg.norm(neighbors - glowworm, axis=1)
                probabilities = luciferin[luciferin > luciferin[i]] / distances
                probabilities /= np.sum(probabilities)
                selected_neighbor = neighbors[
                    np.random.default_rng(self.seed).choice(
                        len(neighbors), p=probabilities
                    )
                ]
                direction = selected_neighbor - glowworm
                random_vector = np.random.default_rng(self.seed).uniform(
                    -1, 1, self.dim
                )
                glowworm += self.step_size * (
                    direction + self.randomness * random_vector
                )
            new_population.append(glowworm)
        return np.array(new_population)

    def search(self) -> tuple[np.ndarray, float]:
        """Run the glowworm swarm optimization algorithm and return the best solution and fitness.

        Returns:
        tuple[np.ndarray, float]: The best solution found by the algorithm and its corresponding fitness value.

        """
        population = self._initialize()
        best_solution = None
        best_fitness = np.inf
        for _ in range(self.max_iter):
            fitness = self._compute_fitness(population)
            luciferin = self._update_luciferin(population, fitness)
            population = self._move_glowworms(population, luciferin)
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < best_fitness:
                best_fitness = fitness[min_fitness_idx]
                best_solution = population[min_fitness_idx]
        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(GlowwormSwarmOptimization)
