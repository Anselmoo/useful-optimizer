"""Tunicate Swarm Algorithm (TSA) implementation.

This module implements the Tunicate Swarm Algorithm, a bio-inspired
optimization algorithm based on the swarm behavior of tunicates
(sea squirts) during navigation and foraging.

Reference:
    Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate
    Swarm Algorithm: A new bio-inspired based metaheuristic paradigm for
    global optimization. Engineering Applications of Artificial Intelligence,
    90, 103541.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_P_MIN = 1  # Minimum parameter for velocity
_P_MAX = 4  # Maximum parameter for velocity


class TunicateSwarmAlgorithm(AbstractOptimizer):
    """Tunicate Swarm Algorithm optimizer.

    The TSA mimics tunicate swarm behavior:
    - Jet propulsion for movement
    - Swarm intelligence for coordination
    - Social forces (avoiding conflict, moving toward food, staying close)

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of tunicates (solutions).


    Example:
        >>> from opt.swarm_intelligence.tunicate_swarm import TunicateSwarmAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = TunicateSwarmAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = TunicateSwarmAlgorithm(
        ...     func=shifted_ackley, dim=2,
        ...     lower_bound=-2.768, upper_bound=2.768,
        ...     max_iter=10, seed=42
        ... )
        >>> _, fitness = optimizer.search()
        >>> isinstance(float(fitness), float)
        True
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
        """Initialize the Tunicate Swarm Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of tunicates (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Tunicate Swarm Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best solution (food source)
        best_idx = np.argmin(fitness)
        food_source = population[best_idx].copy()
        food_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Calculate c values for social forces
            c1 = 2 - iteration * (2 / self.max_iter)  # Decreases from 2 to 0

            for i in range(self.population_size):
                # Random parameters
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()

                # Calculate A (avoiding conflict)
                a = (c1 / _P_MAX) + (_P_MIN / _P_MAX)

                # Calculate c2 and c3 (moving toward and staying close to food)
                c2 = np.random.rand()
                c3 = np.random.rand()

                if r2 >= 0.5:
                    new_position = food_source + a * np.abs(
                        food_source - c2 * population[i]
                    )
                else:
                    new_position = food_source - a * np.abs(
                        food_source - c2 * population[i]
                    )

                # Apply swarm update (average with previous tunicate)
                if i > 0:
                    new_position = (new_position + population[i - 1]) / 2

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < food_fitness:
                        food_source = new_position.copy()
                        food_fitness = new_fitness

        return food_source, food_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(TunicateSwarmAlgorithm)
