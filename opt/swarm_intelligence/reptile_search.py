"""Reptile Search Algorithm (RSA) implementation.

This module implements the Reptile Search Algorithm, a nature-inspired
optimization algorithm based on the hunting behavior of crocodiles.

Reference:
    Abualigah, L., Abd Elaziz, M., Sumari, P., Geem, Z. W., & Gandomi, A. H.
    (2022). Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic
    optimizer. Expert Systems with Applications, 191, 116158.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_ALPHA = 0.1  # Sensitivity parameter
_BETA = 0.1  # Beta parameter for probability


class ReptileSearchAlgorithm(AbstractOptimizer):
    """Reptile Search Algorithm optimizer.

    The RSA mimics crocodile hunting behavior:
    - Encircling prey (high walking)
    - Walking toward prey
    - Hunting coordination
    - Hunting cooperation

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of crocodiles (solutions).


    Example:
        >>> from opt.swarm_intelligence.reptile_search import ReptileSearchAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ReptileSearchAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ReptileSearchAlgorithm(
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
        """Initialize the Reptile Search Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of crocodiles (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Reptile Search Algorithm.

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
            t = iteration / self.max_iter

            # Evolutionary Sense (ES) - decreases over time
            es = 2 * t * (1 - t)

            for i in range(self.population_size):
                r1 = np.random.rand()
                r2 = np.random.rand()

                # Select random solution
                rand_idx = np.random.randint(self.population_size)
                rand_sol = population[rand_idx]

                if t <= 0.25:
                    # High walking (encircling)
                    new_position = best_solution + _ALPHA * es * (
                        rand_sol - population[i]
                    )

                elif t <= 0.5:
                    # Walking toward prey
                    new_position = best_solution * rand_sol * _ALPHA * r1 + (
                        (self.upper_bound - self.lower_bound) * r2 + self.lower_bound
                    ) * (1 - _ALPHA)

                elif t <= 0.75:
                    # Hunting coordination
                    reduce_factor = 2 * es * r1 - es
                    new_position = (
                        best_solution * reduce_factor + rand_sol * reduce_factor * r2
                    )

                else:
                    # Hunting cooperation
                    reduce_factor = 2 * es * r1 - es
                    new_position = best_solution - (
                        reduce_factor * (r2 * best_solution - rand_sol)
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ReptileSearchAlgorithm)
