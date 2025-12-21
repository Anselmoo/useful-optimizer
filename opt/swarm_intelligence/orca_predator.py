"""Orca Predator Algorithm.

Implementation based on:
Jiang, N., Wang, W., Yin, Z., Li, Y. & Zhao, S. (2022).
Orca Predation Algorithm: A new bio-inspired optimizer
for engineering optimization problems.
Expert Systems with Applications, 209, 118321.

The algorithm mimics the hunting strategies of orca whales,
combining carousel feeding and wave-wash feeding techniques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class OrcaPredatorAlgorithm(AbstractOptimizer):
    """Orca Predator Algorithm.

    Simulates orca hunting strategies including:
    - Carousel feeding: Surrounding and herding prey
    - Wave-wash feeding: Creating waves to dislodge prey

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of orcas.
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
        """Execute the Orca Predator Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize orca positions
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Best solution (prey position)
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Linearly decreasing parameter
            a = 2 - 2 * (iteration / self.max_iter)

            for i in range(self.population_size):
                # Coefficient vectors
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2

                # Random position for exploration
                p = np.random.rand()

                if p < 0.5:
                    # Carousel feeding (exploitation)
                    if np.abs(A) < 1:
                        # Encircling prey
                        D = np.abs(C * best_solution - positions[i])
                        new_position = best_solution - A * D
                    else:
                        # Search for prey (exploration)
                        rand_idx = np.random.randint(self.population_size)
                        rand_orca = positions[rand_idx]
                        D = np.abs(C * rand_orca - positions[i])
                        new_position = rand_orca - A * D
                else:
                    # Wave-wash feeding (spiral attack)
                    b = 1.0  # Spiral constant
                    l = np.random.uniform(-1, 1)

                    # Distance to prey
                    distance = np.abs(best_solution - positions[i])

                    # Spiral position update
                    new_position = (
                        distance * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

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
    from opt.demo import run_demo

    run_demo(OrcaPredatorAlgorithm)
