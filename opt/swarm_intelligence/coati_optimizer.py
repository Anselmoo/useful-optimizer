"""Coati Optimization Algorithm.

Implementation based on:
Dehghani, M., Montazeri, Z., Trojovská, E. & Trojovský, P. (2023).
Coati Optimization Algorithm: A new bio-inspired metaheuristic algorithm
for solving optimization problems.
Knowledge-Based Systems, 259, 110011.

The algorithm mimics the hunting strategies of coatis, including
cooperative hunting and foraging behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class CoatiOptimizer(AbstractOptimizer):
    """Coati Optimization Algorithm.

    Simulates the cooperative hunting behavior of coatis including:
    - Chasing iguana on tree: Coordinated group hunting
    - Escaping from predator: Evasive behavior and exploration

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of coatis.
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
        """Execute the Coati Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize coati positions
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Best solution (iguana position)
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Half population uses each strategy
            half_pop = self.population_size // 2

            for i in range(self.population_size):
                if i < half_pop:
                    # Strategy 1: Chasing iguana (exploitation)
                    # Coatis climb tree toward iguana

                    r = np.random.rand()

                    # Iguana tries to escape to random position
                    iguana_pos = best_solution + (
                        (2 * np.random.rand(self.dim) - 1)
                        * (1 - iteration / self.max_iter)
                        * best_solution
                    )
                    iguana_pos = np.clip(iguana_pos, self.lower_bound, self.upper_bound)

                    if r < 0.5:
                        # Move toward iguana on tree
                        new_position = positions[i] + np.random.rand() * (
                            iguana_pos - 2 * np.random.rand() * positions[i]
                        )
                    else:
                        # Move toward iguana on ground
                        new_position = positions[i] + np.random.rand() * (
                            iguana_pos - positions[i]
                        )
                else:
                    # Strategy 2: Escaping from predator (exploration)
                    # Coatis run randomly when predator approaches

                    # Select random coati as reference
                    rand_idx = np.random.randint(self.population_size)
                    rand_coati = positions[rand_idx]

                    # Escape behavior
                    r1, r2 = np.random.rand(2)

                    if fitness[rand_idx] < fitness[i]:
                        # Move toward better coati
                        new_position = positions[i] + r1 * (
                            rand_coati - r2 * positions[i]
                        )
                    else:
                        # Move away from worse coati
                        new_position = positions[i] + r1 * (
                            positions[i] - r2 * rand_coati
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
    from opt.benchmark.functions import shifted_ackley

    optimizer = CoatiOptimizer(
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
