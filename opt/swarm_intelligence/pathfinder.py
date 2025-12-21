"""Pathfinder Algorithm (PFA) implementation.

This module implements the Pathfinder Algorithm, a swarm-based
metaheuristic optimization algorithm inspired by the collective
movement of animal groups searching for food.

Reference:
    Yapici, H., & Cetinkaya, N. (2019). A new meta-heuristic optimizer:
    Pathfinder algorithm. Applied Soft Computing, 78, 545-568.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_ALPHA = 1.0  # Step size coefficient
_BETA = 2.0  # Attraction coefficient


class PathfinderAlgorithm(AbstractOptimizer):
    """Pathfinder Algorithm optimizer.

    The PFA simulates the behavior of animal groups where:
    - A pathfinder (leader) guides the group toward food
    - Members follow the pathfinder with random movement
    - The group adapts to find optimal locations

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of solutions in the population.
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
        """Initialize the Pathfinder Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of solutions in the population.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Pathfinder Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find pathfinder (best solution)
        best_idx = np.argmin(fitness)
        pathfinder = population[best_idx].copy()
        pathfinder_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Update parameters
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Update pathfinder position (exploration)
            update_vec = (
                _ALPHA * np.random.randn(self.dim) * (1 - iteration / self.max_iter)
            )
            new_pathfinder = pathfinder + update_vec

            # Boundary handling for pathfinder
            new_pathfinder = np.clip(new_pathfinder, self.lower_bound, self.upper_bound)

            # Evaluate new pathfinder
            new_fitness = self.func(new_pathfinder)

            if new_fitness < pathfinder_fitness:
                pathfinder = new_pathfinder
                pathfinder_fitness = new_fitness

            # Update member positions
            for i in range(self.population_size):
                if i == best_idx:
                    continue

                # Distance vectors
                d1 = np.abs(pathfinder - population[i])
                d2 = np.abs(population[best_idx] - population[i])

                # Position update
                r = np.random.rand(self.dim)
                epsilon = (1 - iteration / self.max_iter) * np.random.randn(self.dim)

                new_position = (
                    population[i]
                    + r1 * r * (pathfinder - population[i])
                    + r2 * (1 - r) * (population[best_idx] - population[i])
                    + _BETA * epsilon * d1
                )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update if better
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < pathfinder_fitness:
                        pathfinder = new_position.copy()
                        pathfinder_fitness = new_fitness
                        best_idx = i

        return pathfinder, pathfinder_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(PathfinderAlgorithm)
