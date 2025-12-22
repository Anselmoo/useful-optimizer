"""Snow Geese Optimization Algorithm (SGOA).

This module implements the Snow Geese Optimization Algorithm, a swarm
intelligence algorithm inspired by the migration behavior of snow geese.

Snow geese migrate in large flocks following V-formation patterns,
with leaders guiding the flock and rotation of positions for energy efficiency.

Reference:
    Jiang, H., Yang, Y., Ping, W., & Dong, Y. (2023).
    A novel hybrid algorithm based on Snow Geese and Differential Evolution
    for global optimization.
    Applied Soft Computing, 139, 110235.
    DOI: 10.1016/j.asoc.2023.110235

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = SnowGeeseOptimizer(
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


class SnowGeeseOptimizer(AbstractOptimizer):
    """Snow Geese Optimization Algorithm optimizer.

    This algorithm simulates snow geese migration behaviors:
    1. V-formation flying - following leaders efficiently
    2. Leader rotation - changing leader positions
    3. Food search during migration - local exploration

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of geese in the flock.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.snow_geese import SnowGeeseOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SnowGeeseOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SnowGeeseOptimizer(
        ...     func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=2.768, max_iter=10
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
        """Initialize Snow Geese Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of geese. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Snow Geese Optimization Algorithm.

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
            t = iteration / self.max_iter

            # Sort population to determine V-formation positions
            sorted_indices = np.argsort(fitness)
            leader = population[sorted_indices[0]]  # Best goose leads

            for i in range(self.population_size):
                r = np.random.random()

                if r < 0.5:
                    # Phase 1: V-formation flying (exploitation)
                    # Follow the leader with slipstream effect
                    r1 = np.random.random(self.dim)
                    r2 = np.random.random()

                    # Position in formation affects movement
                    rank = np.where(sorted_indices == i)[0][0]
                    formation_factor = 1 - rank / self.population_size

                    new_position = (
                        population[i]
                        + r1 * formation_factor * (leader - population[i])
                        + r2 * (1 - t) * (best_solution - population[i])
                    )

                else:
                    # Phase 2: Food search during migration (exploration)
                    # Geese search for food during rest stops
                    r3 = np.random.random(self.dim)
                    r4 = np.random.random()

                    # Random search with Levy-like movement
                    step_size = (
                        (self.upper_bound - self.lower_bound) * (1 - t) ** 2 * 0.1
                    )

                    # Choose random neighbor to follow
                    neighbor_idx = np.random.randint(self.population_size)
                    neighbor = population[neighbor_idx]

                    new_position = (
                        population[i]
                        + r3 * (neighbor - population[i])
                        + r4 * step_size * np.random.standard_normal(self.dim)
                    )

                # Leader rotation mechanism
                if np.random.random() < 0.1 * (1 - t):  # Decreasing rotation rate
                    # Random perturbation to simulate leader change
                    perturbation = (
                        np.random.standard_normal(self.dim)
                        * (self.upper_bound - self.lower_bound)
                        * 0.05
                        * (1 - t)
                    )
                    new_position = new_position + perturbation

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
    from opt.demo import run_demo

    run_demo(SnowGeeseOptimizer)
