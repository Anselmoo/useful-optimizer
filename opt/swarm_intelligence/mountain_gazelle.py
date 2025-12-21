"""Mountain Gazelle Optimizer.

Implementation based on:
Abdollahzadeh, B., Gharehchopogh, F.S., Khodadadi, N. & Mirjalili, S. (2022).
Mountain Gazelle Optimizer: A new Nature-inspired Metaheuristic Algorithm
for Global Optimization Problems.
Advances in Engineering Software, 174, 103282.

The algorithm mimics the social and territorial behaviors of mountain gazelles,
including grazing, mating, and avoiding predators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class MountainGazelleOptimizer(AbstractOptimizer):
    """Mountain Gazelle Optimizer.

    Simulates the behavior of mountain gazelles including:
    - Grazing: Searching for food in territory
    - Fighting: Competition between males
    - Fear from predators: Escape behavior

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of gazelles.
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
        """Execute the Mountain Gazelle Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize gazelle positions
        positions = np.random.uniform(
            self.lower_bound,
            self.upper_bound,
            (self.population_size, self.dim),
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Best solution
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Top gazelles (elite)
        n_elite = max(3, self.population_size // 5)

        for iteration in range(self.max_iter):
            # Coefficients update
            a = 2 * (1 - (iteration / self.max_iter) ** 2)  # Decreases from 2 to 0

            # Sort to find elite gazelles
            sorted_idx = np.argsort(fitness)
            elite_positions = positions[sorted_idx[:n_elite]]

            for i in range(self.population_size):
                r = np.random.rand()

                # Select random elite member
                elite_idx = np.random.randint(n_elite)
                elite = elite_positions[elite_idx]

                if r < 1 / 3:
                    # Grazing behavior - exploration
                    r1, r2 = np.random.rand(2)
                    A = a * (2 * r1 - 1)
                    C = 2 * r2

                    # Random gazelle
                    rand_idx = np.random.randint(self.population_size)
                    rand_gazelle = positions[rand_idx]

                    new_position = (
                        positions[i]
                        + A * (rand_gazelle - positions[i])
                        + C * (elite - positions[i])
                    )

                elif r < 2 / 3:
                    # Fighting behavior - competition with elite
                    r3 = np.random.rand()

                    # Male fights around the female (elite)
                    new_position = (
                        elite
                        + (2 * r3 - 1) * a * (elite - positions[i])
                    )

                else:
                    # Fear from predators - escape behavior
                    r4 = np.random.rand()

                    # Random step away from predator (worst solution)
                    worst_idx = sorted_idx[-1]
                    worst = positions[worst_idx]

                    # Escape direction
                    escape_direction = positions[i] - worst
                    escape_direction = escape_direction / (
                        np.linalg.norm(escape_direction) + 1e-10
                    )

                    new_position = (
                        positions[i]
                        + r4 * a * escape_direction
                        * (self.upper_bound - self.lower_bound)
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

    optimizer = MountainGazelleOptimizer(
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
