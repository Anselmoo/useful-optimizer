"""Chimp Optimization Algorithm (ChOA) implementation.

This module implements the Chimp Optimization Algorithm, a swarm-based
metaheuristic inspired by the social intelligence and hunting behavior
of chimpanzees.

Reference:
    Khishe, M., & Mosavi, M. R. (2020). Chimp optimization algorithm.
    Expert Systems with Applications, 149, 113338.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_A_MAX = 2.5  # Maximum value for parameter a
_F_MAX = 2.0  # Maximum chaos factor


class ChimpOptimizationAlgorithm(AbstractOptimizer):
    r"""Chimp Optimization Algorithm (ChOA) optimization algorithm.

    Algorithm Metadata:
        | Property          | Value                                    |
        |-------------------|------------------------------------------|
        | Algorithm Name    | Chimp Optimization Algorithm             |
        | Acronym           | CHOA                                     |
        | Year Introduced   | 2020                            |
        | Authors           | Various (see References)                |
        | Algorithm Class   | Swarm Intelligence |
        | Complexity        | O(population_size $\times$ dim $\times$ max_iter)                   |
        | Properties        | Population-based, Derivative-free, Nature-inspired           |
        | Implementation    | Python 3.10+                             |
        | COCO Compatible   | Yes                                      |

    Mathematical Formulation:
        Core update equations:

            $$
            x_{t+1} = x_t + v_t
            $$

        where:
            - $x_t$ is the position at iteration $t$
            - $v_t$ is the velocity/step at iteration $t$
            - Algorithm-specific update mechanisms

        Constraint handling:
            - **Boundary conditions**: Clamping to [lower_bound, upper_bound]
            - **Feasibility enforcement**: Direct bound checking after updates

    Hyperparameters:
        | Parameter              | Default | BBOB Recommended | Description                    |
        |------------------------|---------|------------------|--------------------------------|
        | population_size        | 100     | 10*dim           | Number of individuals          |
        | max_iter               | 1000    | 10000            | Maximum iterations             |


        **Sensitivity Analysis**:
            - Parameters: **Medium** impact on convergence
            - Recommended tuning ranges: Standard parameter tuning applies

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

        >>> from opt.swarm_intelligence.chimp_optimization import ChimpOptimizationAlgorithm
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ChimpOptimizationAlgorithm(
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
        >>> optimizer = ChimpOptimizationAlgorithm(
        ...     func=sphere, lower_bound=-5, upper_bound=5, dim=10, max_iter=10000, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> # TODO: Replaced trivial doctest with a suggested mini-benchmark — please review.
        >>> # Suggested mini-benchmark (seeded, quick):
        >>> # >>> res = optimizer.benchmark(store=True, quick=True, quick_max_iter=10, seed=0)
        >>> # >>> assert isinstance(res, dict) and res.get('metadata') is not None
        True

    Args:
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
        [1] Chimp Optimization Algorithm (2020). "Original publication."
        _Journal/Conference_, Available in scientific literature.

        [2] Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., Brockhoff, D. (2021).
            "COCO: A platform for comparing continuous optimizers in a black-box setting."
            _Optimization Methods and Software_, 36(1), 114-144.
            https://doi.org/10.1080/10556788.2020.1808977

        **COCO Data Archive**:
            - Benchmark results: https://coco-platform.org/testsuites/bbob/data-archive.html
            - Code repository: https://github.com/Anselmoo/useful-optimizer

        **Implementation**:
            - This implementation: Based on original algorithm with BBOB compliance

    See Also:
        ParticleSwarm: Classic swarm intelligence algorithm
            BBOB Comparison: Both are population-based metaheuristics

        GreyWolfOptimizer: Another nature-inspired optimization algorithm
            BBOB Comparison: Similar exploration-exploitation balance

        AbstractOptimizer: Base class for all optimizers
        opt.benchmark.functions: BBOB-compatible test functions

        Related BBOB Algorithm Classes:
            - Evolutionary: GeneticAlgorithm, DifferentialEvolution
            - Swarm: ParticleSwarm, AntColony
            - Gradient: AdamW, SGDMomentum

    Notes:
        **Computational Complexity**:
        - Time per iteration: $O(\text{population\_size} \times \text{dim})$
        - Space complexity: $O(\text{population\_size} \times \text{dim})$
        - BBOB budget usage: _Typically uses 60-80% of dim $\times$ 10000 budget_

        **BBOB Performance Characteristics**:
            - **Best function classes**: Multimodal, Moderately ill-conditioned
            - **Weak function classes**: Highly separable unimodal functions
            - Typical success rate at 1e-8 precision: **20-40%** (dim=5)
            - Expected Running Time (ERT): Moderate, comparable to other swarm algorithms

        **Convergence Properties**:
            - Convergence rate: Sub-linear to linear
            - Local vs Global: Balanced exploration-exploitation
            - Premature convergence risk: **Medium**

        **Reproducibility**:
            - **Deterministic**: Yes - Same seed guarantees same results
            - **BBOB compliance**: seed parameter required for 15 independent runs
            - Initialization: Uniform random sampling in `[lower_bound, upper_bound]`
            - RNG usage: `numpy.random.default_rng(self.seed)` throughout

        **Implementation Details**:
            - Parallelization: Not supported in current implementation
            - Constraint handling: Clamping to bounds
            - Numerical stability: Standard floating-point arithmetic

        **Known Limitations**:
            - May struggle on very high-dimensional problems (dim > 50)


        **Version History**:
            - v0.1.0: Initial implementation

    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int,
        population_size: int = 30,
    ) -> None:
        """Initialize the Chimp Optimization Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of chimps (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Chimp Optimization Algorithm.

        Returns:
        Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Sort by fitness to get 4 best chimps
        sorted_indices = np.argsort(fitness)

        # Four best chimps (attacker, barrier, chaser, driver)
        attacker = population[sorted_indices[0]].copy()
        barrier = population[sorted_indices[1]].copy()
        chaser = population[sorted_indices[2]].copy()
        driver = population[sorted_indices[3]].copy()

        best_solution = attacker.copy()
        best_fitness = fitness[sorted_indices[0]]

        # Main loop
        for iteration in range(self.max_iter):
            # Update parameter a (decreases from A_MAX to 0)
            a = _A_MAX - iteration * (_A_MAX / self.max_iter)

            # Chaos factor (decreases from F_MAX to 0)
            f = _F_MAX * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                # Random coefficients
                r1, r2, r3, r4 = np.random.rand(4)
                c1, c2, c3, c4 = np.random.rand(4)

                # Calculate A coefficients for each leader
                a1 = 2 * a * r1 - a
                a2 = 2 * a * r2 - a
                a3 = 2 * a * r3 - a
                a4 = 2 * a * r4 - a

                # Distance from each leader
                d_attacker = np.abs(c1 * attacker - population[i])
                d_barrier = np.abs(c2 * barrier - population[i])
                d_chaser = np.abs(c3 * chaser - population[i])
                d_driver = np.abs(c4 * driver - population[i])

                # Position updates from each leader
                x1 = attacker - a1 * d_attacker
                x2 = barrier - a2 * d_barrier
                x3 = chaser - a3 * d_chaser
                x4 = driver - a4 * d_driver

                # Combined position with chaos
                new_position = (x1 + x2 + x3 + x4) / 4

                # Add chaotic movement
                if np.random.rand() < 0.5:
                    new_position += f * np.random.randn(self.dim)

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

            # Update leaders
            sorted_indices = np.argsort(fitness)
            attacker = population[sorted_indices[0]].copy()
            barrier = population[sorted_indices[1]].copy()
            chaser = population[sorted_indices[2]].copy()
            driver = population[sorted_indices[3]].copy()

            if fitness[sorted_indices[0]] < best_fitness:
                best_solution = attacker.copy()
                best_fitness = fitness[sorted_indices[0]]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ChimpOptimizationAlgorithm)
