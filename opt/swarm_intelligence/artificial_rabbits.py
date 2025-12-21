"""Artificial Rabbits Optimization (ARO) Algorithm.

This module implements the Artificial Rabbits Optimization algorithm,
a bio-inspired metaheuristic based on the survival strategies of rabbits.

Rabbits exhibit two main survival behaviors: detour foraging (moving
irregularly to avoid predators) and random hiding (seeking shelter).

Reference:
    Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022).
    Artificial rabbits optimization: A new bio-inspired meta-heuristic
    algorithm for solving engineering optimization problems.
    Engineering Applications of Artificial Intelligence, 114, 105082.
    DOI: 10.1016/j.engappai.2022.105082

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = ArtificialRabbitsOptimizer(
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


class ArtificialRabbitsOptimizer(AbstractOptimizer):
    """Artificial Rabbits Optimization algorithm optimizer.

    This algorithm simulates rabbit survival behaviors:
    1. Detour foraging - exploration with random movements
    2. Random hiding - exploitation toward hiding locations
    3. Energy-based transitions between phases

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of rabbits in the population.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.artificial_rabbits import ArtificialRabbitsOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ArtificialRabbitsOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ArtificialRabbitsOptimizer(
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
        population_size: int = 30,
        max_iter: int = 100,
    ) -> None:
        """Initialize Artificial Rabbits Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of rabbits. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Artificial Rabbits Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Energy factor decreases linearly
            a = 4 * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                # Generate random vector for detour foraging
                r1 = np.random.random()
                r2 = np.random.random()
                r3 = np.random.random()

                # Random rabbit selection
                l_idx = np.random.randint(self.population_size)
                while l_idx == i:
                    l_idx = np.random.randint(self.population_size)

                # Create random binary mask for dimension selection
                r_mask = np.random.random(self.dim) < 0.5

                if r1 < 0.5:
                    # Detour foraging strategy (exploration)
                    # Random perturbation based on another rabbit
                    c = np.random.randint(1, self.dim + 1)
                    random_dims = np.random.choice(self.dim, c, replace=False)

                    new_position = population[i].copy()
                    for j in random_dims:
                        g = np.random.standard_normal()
                        new_position[j] = population[l_idx][j] + a * g * (
                            population[i][j] - population[l_idx][j]
                        )
                else:
                    # Random hiding strategy (exploitation)
                    # Move toward hiding burrow near best position
                    h = (
                        (self.max_iter - iteration + 1)
                        / self.max_iter
                        * np.random.standard_normal()
                    )

                    # Create hiding position
                    hiding_burrow = best_solution + h * (
                        r2 * best_solution - r3 * population[i]
                    )

                    new_position = np.where(r_mask, hiding_burrow, population[i])

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
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

    optimizer = ArtificialRabbitsOptimizer(
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
