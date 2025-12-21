"""Giant Trevally Optimizer (GTO).

This module implements the Giant Trevally Optimizer, a bio-inspired
metaheuristic algorithm based on the hunting behavior of giant trevally fish.

Giant trevallies are apex predators known for their remarkable hunting
strategy of jumping out of water to catch birds and cooperative hunting.

Reference:
    Sadeeq, H. T., & Abdulazeez, A. M. (2022).
    Giant Trevally Optimizer (GTO): A Novel Metaheuristic Algorithm for
    Global Optimization and Challenging Engineering Problems.
    IEEE Access, 10, 121615-121640.
    DOI: 10.1109/ACCESS.2022.3223388

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = GiantTrevallyOptimizer(
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


class GiantTrevallyOptimizer(AbstractOptimizer):
    """Giant Trevally Optimizer algorithm.

    This algorithm simulates giant trevally hunting behaviors:
    1. Foraging movement - searching for prey underwater
    2. Jump and catch - explosive attack on prey (birds)
    3. Cooperative hunting - group hunting strategies

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of fish in the school.
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
        """Initialize Giant Trevally Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of fish. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Giant Trevally Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize school of fish
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            t = iteration / self.max_iter

            for i in range(self.population_size):
                r = np.random.random()

                if r < 0.5:
                    # Phase 1: Foraging movement (exploration)
                    # Fish searching for prey underwater
                    step = (
                        np.random.standard_normal(self.dim)
                        * (self.upper_bound - self.lower_bound)
                        * (1 - t)
                    )

                    # Random exploration with decreasing range
                    new_position = population[i] + step * 0.1

                else:
                    # Phase 2: Jump and catch (exploitation)
                    # Fish jumping to catch prey near best position
                    r1 = np.random.random(self.dim)
                    r2 = 2 * np.random.random() - 1  # [-1, 1]

                    # Exponential jump factor
                    jump_factor = np.exp(-4 * t)  # Decreases over time

                    # Jump toward best solution
                    new_position = (
                        best_solution
                        + r1 * jump_factor * (best_solution - population[i])
                        + r2
                        * (1 - t)
                        * np.random.standard_normal(self.dim)
                        * 0.01
                        * (self.upper_bound - self.lower_bound)
                    )

                # Cooperative hunting enhancement
                if np.random.random() < 0.1:  # 10% chance of cooperation
                    partner_idx = np.random.randint(self.population_size)
                    if fitness[partner_idx] < fitness[i]:
                        r3 = np.random.random(self.dim)
                        new_position = (
                            new_position
                            + r3 * (population[partner_idx] - new_position) * 0.5
                        )

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

    optimizer = GiantTrevallyOptimizer(
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
