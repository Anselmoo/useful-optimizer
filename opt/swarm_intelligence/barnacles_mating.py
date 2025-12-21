"""Barnacles Mating Optimizer.

Implementation based on:
Sulaiman, M.H., Mustaffa, Z., Saari, M.M. & Daniyal, H. (2020).
Barnacles Mating Optimizer: A new bio-inspired algorithm for solving
engineering optimization problems.
Engineering Applications of Artificial Intelligence, 87, 103330.

The algorithm mimics the mating behavior of barnacles, where sessile
creatures must extend their reproductive organs to reach nearby mates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Plenum constant
_PL = 4  # Selection parameter


class BarnaclesMatingOptimizer(AbstractOptimizer):
    """Barnacles Mating Optimizer algorithm.

    Mimics the unique mating behavior of barnacles, which are sessile
    crustaceans that extend their penis (the longest relative to body
    size in the animal kingdom) to fertilize nearby mates.

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of barnacles.
        pl: Plenum constant controlling selection pressure. Default 4.


    Example:
        >>> from opt.swarm_intelligence.barnacles_mating import BarnaclesMatingOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = BarnaclesMatingOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = BarnaclesMatingOptimizer(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10
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
        pl: int = _PL,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.pl = pl

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Barnacles Mating Optimizer algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize barnacle positions
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Find best position
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Update temperature linearly decreasing
            temperature = 1 - (iteration / self.max_iter)

            for i in range(self.population_size):
                # Select father and mother barnacles using tournament selection
                candidates = np.random.choice(
                    self.population_size, size=self.pl, replace=False
                )
                candidate_fitness = fitness[candidates]
                sorted_candidates = candidates[np.argsort(candidate_fitness)]
                father_idx = sorted_candidates[0]
                mother_idx = sorted_candidates[1]

                father = positions[father_idx]
                mother = positions[mother_idx]

                # Generate offspring
                new_position = np.zeros(self.dim)

                for d in range(self.dim):
                    p = np.random.rand()

                    if p < temperature:
                        # Hardy-Weinberg approach - select genes from parents
                        q = np.random.rand()
                        new_position[d] = q * father[d] + (1 - q) * mother[d]
                    else:
                        # Sperm cast - random selection from population
                        rand_idx = np.random.randint(self.population_size)
                        new_position[d] = positions[rand_idx, d]

                # Apply boundary constraints
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(BarnaclesMatingOptimizer)
