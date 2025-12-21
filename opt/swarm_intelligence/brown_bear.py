"""Brown Bear Optimization Algorithm.

Implementation based on:
Prakash, T., Singh, P.P., Singh, V.P. & Singh, S.N. (2023).
A Novel Brown-bear Optimization Algorithm for Solving Economic Dispatch
Problem.
In Advanced Computing and Intelligent Technologies (pp. 137-148).

The algorithm mimics the foraging and hunting behaviors of brown bears
in search of food sources like salmon and berries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class BrownBearOptimizer(AbstractOptimizer):
    """Brown Bear Optimization Algorithm.

    Simulates the foraging behavior of brown bears including:
    - Pedal marking: Territory establishment
    - Sniffing: Searching for food sources
    - Chasing: Pursuing prey (exploitation)

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of bears.
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
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Brown Bear Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize bear positions
        positions = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.population_size, self.dim),
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Best solution (best food source)
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Exploration-exploitation balance
            w = 0.5 * (1 - iteration / self.max_iter)  # Decreasing weight

            for i in range(self.population_size):
                r = np.random.rand()

                if r < 0.5:
                    # Pedal marking and sniffing (exploration)
                    # Bears explore territory randomly

                    # Select random bears for interaction
                    r1, r2 = np.random.choice(
                        self.population_size, size=2, replace=False
                    )
                    bear1, bear2 = positions[r1], positions[r2]

                    # Random exploration with marking behavior
                    rand_factor = np.random.randn(self.dim)
                    new_position = (
                        positions[i]
                        + w * rand_factor * (bear1 - bear2)
                    )
                else:
                    # Chasing behavior (exploitation)
                    # Bears chase toward best food source

                    # Intensity decreases with iterations
                    intensity = 2 * (1 - iteration / self.max_iter)

                    # Random chase parameters
                    r3, r4 = np.random.rand(2)

                    if r3 < 0.5:
                        # Direct chase
                        new_position = (
                            best_solution
                            - intensity * r4 * (best_solution - positions[i])
                        )
                    else:
                        # Circular chase (spiral)
                        angle = 2 * np.pi * r4
                        distance = np.abs(best_solution - positions[i])
                        new_position = (
                            best_solution
                            + distance * np.cos(angle)
                            * (1 - iteration / self.max_iter)
                        )

                # Boundary handling
                new_position = np.clip(
                    new_position, self.lower_bound, self.upper_bound
                )

                # Evaluate new position
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = BrownBearOptimizer(
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
