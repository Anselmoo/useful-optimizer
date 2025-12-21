"""Pelican Optimization Algorithm (POA).

This module implements the Pelican Optimization Algorithm, a bio-inspired
metaheuristic based on the hunting behavior of pelicans.

Pelicans are known for their cooperative hunting strategies, including
group fishing and synchronized diving to catch prey.

Reference:
    Trojovský, P., & Dehghani, M. (2022).
    Pelican Optimization Algorithm: A Novel Nature-Inspired Algorithm for
    Engineering Applications.
    Sensors, 22(3), 855.
    DOI: 10.3390/s22030855

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = PelicanOptimizer(
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


class PelicanOptimizer(AbstractOptimizer):
    """Pelican Optimization Algorithm optimizer.

    This algorithm simulates pelican hunting behaviors:
    1. Moving toward prey - pelicans approach fish
    2. Winging on water surface - coordinated fishing
    3. Scoop fishing - diving and catching prey

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of pelicans in the flock.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.pelican_optimizer import PelicanOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = PelicanOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = PelicanOptimizer(
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
        """Initialize Pelican Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of pelicans. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Pelican Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize flock
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)

            for i in range(self.population_size):
                # Phase 1: Moving toward prey (exploration)
                # Select a prey location (random better solution)
                better_indices = sorted_indices[: i + 1] if i > 0 else [0]
                prey_idx = better_indices[np.random.randint(len(better_indices))]
                prey = population[prey_idx]

                r1 = np.random.random(self.dim)
                r2 = np.random.random()

                # Approach prey
                if fitness[prey_idx] < fitness[i]:
                    new_position = population[i] + r1 * (
                        prey - population[i] * (1 + r2)
                    )
                else:
                    new_position = population[i] + r1 * (population[i] - prey)

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                # Phase 2: Winging on water surface (exploitation)
                t = 1 - iteration / self.max_iter
                r3 = np.random.random(self.dim)
                r4 = np.random.random()

                # Coordinated fishing near best solution
                epsilon = 0.001  # Small perturbation
                wing_position = (
                    best_solution
                    + r3 * (best_solution - population[i]) * t
                    + epsilon * (2 * r4 - 1) * (self.upper_bound - self.lower_bound)
                )

                wing_position = np.clip(
                    wing_position, self.lower_bound, self.upper_bound
                )
                wing_fitness = self.func(wing_position)

                if wing_fitness < fitness[i]:
                    population[i] = wing_position
                    fitness[i] = wing_fitness

                # Update best solution
                if fitness[i] < best_fitness:
                    best_solution = population[i].copy()
                    best_fitness = fitness[i]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = PelicanOptimizer(
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
