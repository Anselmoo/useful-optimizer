"""Manta Ray Foraging Optimization (MRFO) implementation.

This module implements the Manta Ray Foraging Optimization algorithm, a
nature-inspired metaheuristic based on the foraging behaviors of manta rays.

Reference:
    Zhao, W., Zhang, Z., & Wang, L. (2020). Manta ray foraging optimization:
    An effective bio-inspired optimizer for engineering applications.
    Engineering Applications of Artificial Intelligence, 87, 103300.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_SOMERSAULT_FACTOR = 2.0  # Somersault range factor


class MantaRayForagingOptimization(AbstractOptimizer):
    """Manta Ray Foraging Optimization algorithm.

    The MRFO mimics three foraging behaviors of manta rays:
    1. Chain foraging - manta rays form a chain to filter plankton
    2. Cyclone foraging - manta rays form a spiral to concentrate prey
    3. Somersault foraging - manta rays somersault to change direction

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of manta rays (solutions).


    Example:
        >>> from opt.swarm_intelligence.manta_ray import MantaRayForagingOptimization
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = MantaRayForagingOptimization(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = MantaRayForagingOptimization(
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
        """Initialize the Manta Ray Foraging Optimization.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of manta rays (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Manta Ray Foraging Optimization.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Calculate coefficient
            coef = iteration / self.max_iter

            for i in range(self.population_size):
                r = np.random.rand()
                r1 = np.random.rand()

                if r < 1.0 / 3.0:
                    # Chain foraging
                    if i == 0:
                        new_position = population[i] + r1 * (
                            best_solution - population[i]
                        )
                    else:
                        new_position = population[i] + r1 * (
                            population[i - 1] - population[i]
                        )

                elif r < 2.0 / 3.0:
                    # Cyclone foraging
                    beta = (
                        2
                        * np.exp(r1 * (self.max_iter - iteration + 1) / self.max_iter)
                        * np.sin(2 * np.pi * r1)
                    )

                    if coef < np.random.rand():
                        # Random position reference
                        rand_idx = np.random.randint(self.population_size)
                        rand_pos = population[rand_idx]

                        if i == 0:
                            new_position = (
                                rand_pos
                                + np.random.rand(self.dim) * (rand_pos - population[i])
                                + beta * (rand_pos - population[i])
                            )
                        else:
                            new_position = (
                                rand_pos
                                + np.random.rand(self.dim)
                                * (population[i - 1] - population[i])
                                + beta * (rand_pos - population[i])
                            )
                    # Best position reference
                    elif i == 0:
                        new_position = (
                            best_solution
                            + np.random.rand(self.dim) * (best_solution - population[i])
                            + beta * (best_solution - population[i])
                        )
                    else:
                        new_position = (
                            best_solution
                            + np.random.rand(self.dim)
                            * (population[i - 1] - population[i])
                            + beta * (best_solution - population[i])
                        )

                else:
                    # Somersault foraging
                    s_factor = _SOMERSAULT_FACTOR
                    new_position = population[i] + s_factor * (
                        np.random.rand() * best_solution
                        - np.random.rand() * population[i]
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new solution
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

    optimizer = MantaRayForagingOptimization(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        max_iter=100,
        population_size=30,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
