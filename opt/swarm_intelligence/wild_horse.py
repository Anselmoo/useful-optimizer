"""Wild Horse Optimizer.

Implementation based on:
Naruei, I. & Keynia, F. (2022).
Wild Horse Optimizer: A new meta-heuristic algorithm for solving
engineering optimization problems.
Engineering with Computers, 38(4), 3025-3056.

The algorithm mimics the social behavior of wild horses including
grazing, fighting, and herd dynamics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_TDR = 0.1  # Team development rate
_PS = 0.5  # Probability of stallion selection


class WildHorseOptimizer(AbstractOptimizer):
    """Wild Horse Optimizer.

    Simulates the social behavior of wild horse groups, including:
    - Grazing behavior for exploration
    - Stallion mating and fighting behavior
    - Group dynamics and leadership changes

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of horses in the population.
        n_groups: Number of horse groups. Default 5.


    Example:
        >>> from opt.swarm_intelligence.wild_horse import WildHorseOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = WildHorseOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = WildHorseOptimizer(
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
        max_iter: int,
        population_size: int = 30,
        n_groups: int = 5,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.n_groups = n_groups
        self.group_size = population_size // n_groups

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Wild Horse Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize horse population
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Sort by fitness and divide into groups
        sorted_indices = np.argsort(fitness)
        positions = positions[sorted_indices]
        fitness = fitness[sorted_indices]

        # Best overall solution
        best_solution = positions[0].copy()
        best_fitness = fitness[0]

        for iteration in range(self.max_iter):
            # Calculate adaptive parameter
            tdr = _TDR * (1 - iteration / self.max_iter)

            # Update each group
            for g in range(self.n_groups):
                start_idx = g * self.group_size
                end_idx = min(start_idx + self.group_size, self.population_size)

                # Stallion is the first (best) in the group
                stallion_idx = start_idx
                stallion = positions[stallion_idx]

                # Update mares (rest of the group)
                for i in range(start_idx + 1, end_idx):
                    r = np.random.rand()

                    if r < _PS:
                        # Grazing behavior - move toward stallion
                        r1, r2 = np.random.rand(2)
                        positions[i] = (
                            2 * r1 * np.cos(2 * np.pi * r2) * (stallion - positions[i])
                            + stallion
                        )
                    else:
                        # Mating behavior
                        # Select random horse from another group
                        other_group = np.random.randint(self.n_groups)
                        while other_group == g:
                            other_group = np.random.randint(self.n_groups)

                        other_start = other_group * self.group_size
                        other_idx = np.random.randint(
                            other_start,
                            min(other_start + self.group_size, self.population_size),
                        )
                        other = positions[other_idx]

                        # Crossover
                        r3 = np.random.rand(self.dim)
                        positions[i] = r3 * positions[i] + (1 - r3) * other

                    # Apply boundary constraints
                    positions[i] = np.clip(
                        positions[i], self.lower_bound, self.upper_bound
                    )

                    # Evaluate new position
                    new_fitness = self.func(positions[i])
                    fitness[i] = new_fitness

            # Leader selection phase
            if np.random.rand() < tdr:
                # Challenge the stallion
                for g in range(self.n_groups):
                    start_idx = g * self.group_size
                    end_idx = min(start_idx + self.group_size, self.population_size)

                    # Find best in group
                    group_fitness = fitness[start_idx:end_idx]
                    best_in_group = start_idx + np.argmin(group_fitness)

                    # Swap if better than stallion
                    if best_in_group != start_idx:
                        positions[[start_idx, best_in_group]] = positions[
                            [best_in_group, start_idx]
                        ]
                        fitness[[start_idx, best_in_group]] = fitness[
                            [best_in_group, start_idx]
                        ]

            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = positions[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(WildHorseOptimizer)
