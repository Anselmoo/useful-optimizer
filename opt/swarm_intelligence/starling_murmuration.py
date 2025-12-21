"""Starling Murmuration Optimizer (SMO).

This module implements the Starling Murmuration Optimizer, a swarm intelligence
algorithm inspired by the collective behavior of starlings during murmuration.

Murmurations are the stunning aerial displays created when thousands of
starlings fly together, creating complex patterns while maintaining cohesion.

Reference:
    Zamani, H., Nadimi-Shahraki, M. H., & Gandomi, A. H. (2022).
    Starling murmuration optimizer: A novel bio-inspired algorithm for
    global and engineering optimization.
    Computer Methods in Applied Mechanics and Engineering, 392, 114616.
    DOI: 10.1016/j.cma.2022.114616

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = StarlingMurmurationOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     population_size=30,
    ...     max_iter=100,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer

if TYPE_CHECKING:
    from collections.abc import Callable


class StarlingMurmurationOptimizer(AbstractOptimizer):
    """Starling Murmuration Optimizer.

    This algorithm simulates starling murmuration behaviors:
    1. Separation - avoiding crowding with nearby birds
    2. Alignment - steering toward average direction of neighbors
    3. Cohesion - moving toward average position of neighbors
    4. Predator avoidance - collective escape maneuvers

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of starlings in the flock.
        max_iter: Maximum number of iterations.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 30,
        max_iter: int = 100,
    ) -> None:
        """Initialize Starling Murmuration Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of starlings. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.neighbor_count = max(3, population_size // 5)  # Topological neighbors

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Starling Murmuration Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize flock
        population = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.population_size, self.dim),
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            t = iteration / self.max_iter

            for i in range(self.population_size):
                # Find topological neighbors (k nearest)
                distances = np.linalg.norm(
                    population - population[i], axis=1
                )
                neighbor_indices = np.argsort(distances)[1 : self.neighbor_count + 1]

                # Calculate center of neighbors
                neighbor_center = np.mean(population[neighbor_indices], axis=0)

                # Random factors
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                r3 = np.random.random()

                if r3 < 0.5:
                    # Cohesion and alignment behavior (exploitation)
                    # Move toward neighbor center and best solution
                    new_position = (
                        population[i]
                        + r1 * (neighbor_center - population[i])
                        + r2 * (1 - t) * (best_solution - population[i])
                    )
                else:
                    # Separation and exploration behavior
                    # Random flight pattern with predator avoidance
                    predator = population[np.argmax(fitness)]  # Worst solution

                    escape_vector = population[i] - predator
                    escape_vector = escape_vector / (
                        np.linalg.norm(escape_vector) + 1e-10
                    )

                    random_step = (
                        np.random.standard_normal(self.dim)
                        * (self.upper_bound - self.lower_bound)
                        * (1 - t)
                        * 0.1
                    )

                    new_position = (
                        population[i]
                        + r1 * escape_vector * (1 - t)
                        + random_step
                    )

                # Boundary handling
                new_position = np.clip(
                    new_position, self.lower_bound, self.upper_bound
                )
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = StarlingMurmurationOptimizer(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        population_size=30,
        max_iter=100,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
