"""Sand Cat Swarm Optimization Algorithm.

Implementation based on:
Seyyedabbasi, A. & Kiani, F. (2023).
Sand Cat swarm optimization: A nature-inspired algorithm to solve
global optimization problems.
Engineering with Computers, 39(4), 2627-2651.

The algorithm mimics the hunting behavior of sand cats, small wild cats
that are efficient hunters in desert environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_R_MAX = 2.0  # Maximum value of sensitivity range
_R_MIN = 0.0  # Minimum value of sensitivity range


class SandCatSwarmOptimizer(AbstractOptimizer):
    """Sand Cat Swarm Optimization Algorithm.

    Simulates the hunting behavior of sand cats, combining:
    - Search mode: Global exploration when prey is far
    - Attack mode: Local exploitation when prey is near

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of sand cats.
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
        """Execute the Sand Cat Swarm Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize sand cat positions
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
            # Update sensitivity range (decreases over iterations)
            r_g = _R_MAX - ((_R_MAX - _R_MIN) * (iteration / self.max_iter) ** 2)

            for i in range(self.population_size):
                # Random parameters
                r = r_g * np.random.rand()
                rand = np.random.rand()

                # Calculate random angle
                theta = np.random.rand() * 2 * np.pi

                if rand < 0.5:
                    # Search mode (exploration)
                    # Select random sand cat
                    rand_idx = np.random.randint(self.population_size)
                    rand_cat = positions[rand_idx]

                    # Update position using spiral movement
                    new_position = r * (
                        rand_cat - np.random.rand() * positions[i]
                    ) + np.abs(np.random.randn(self.dim)) * np.cos(theta)
                else:
                    # Attack mode (exploitation)
                    # Move toward best solution
                    distance = np.abs(best_solution - positions[i])
                    new_position = best_solution - r * distance * np.cos(theta)

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

    run_demo(SandCatSwarmOptimizer)
