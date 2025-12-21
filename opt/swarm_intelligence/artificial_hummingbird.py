"""Artificial Hummingbird Algorithm.

Implementation based on:
Zhao, W., Wang, L. & Mirjalili, S. (2022).
Artificial hummingbird algorithm: A new bio-inspired optimizer with
its engineering applications.
Computer Methods in Applied Mechanics and Engineering, 388, 114194.

The algorithm mimics the unique flight patterns and foraging behavior
of hummingbirds, known for their hovering capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class ArtificialHummingbirdAlgorithm(AbstractOptimizer):
    """Artificial Hummingbird Algorithm.

    Simulates the foraging behavior of hummingbirds including:
    - Guided foraging: Moving toward food sources
    - Territorial foraging: Local exploitation
    - Migration foraging: Global exploration

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of hummingbirds.
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
        """Execute the Artificial Hummingbird Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize hummingbird positions (food sources)
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Visit table (track visits to food sources)
        visit_table = np.ones((self.population_size, self.population_size))
        np.fill_diagonal(visit_table, 0)

        # Best solution
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Calculate direction switch parameter
            dir_switch = 2 * np.random.rand() * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                r = np.random.rand()

                if r < 1 / 3:
                    # Guided foraging - move toward random food source
                    # Select food source based on visit table
                    visit_probs = visit_table[i] / np.sum(visit_table[i])
                    target_idx = np.random.choice(self.population_size, p=visit_probs)
                    target = positions[target_idx]

                    # Flight vector with direction
                    flight_vec = np.random.randn(self.dim)
                    new_position = positions[i] + dir_switch * flight_vec * (
                        target - positions[i]
                    )

                    # Update visit table
                    visit_table[i, target_idx] += 1

                elif r < 2 / 3:
                    # Territorial foraging - local search
                    # Diagonal flight
                    diag_direction = np.random.choice([-1, 1], size=self.dim)
                    step_size = (
                        dir_switch
                        * 0.01
                        * (self.upper_bound - self.lower_bound)
                        * np.random.rand()
                    )
                    new_position = positions[i] + step_size * diag_direction

                else:
                    # Migration foraging - global exploration
                    # Axial flight toward best
                    axis = np.random.randint(self.dim)
                    new_position = positions[i].copy()
                    new_position[axis] = positions[
                        i, axis
                    ] + dir_switch * np.random.randn() * (
                        best_solution[axis] - positions[i, axis]
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

                    # Update best if needed
                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Reset visit table periodically
            if iteration % 10 == 0 and iteration > 0:
                visit_table = np.ones((self.population_size, self.population_size))
                np.fill_diagonal(visit_table, 0)

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = ArtificialHummingbirdAlgorithm(
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
