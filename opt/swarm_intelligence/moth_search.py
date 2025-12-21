"""Moth Search Algorithm.

Implementation based on:
Wang, G.G. (2018).
Moth search algorithm: a bio-inspired metaheuristic algorithm for
global optimization problems.
Memetic Computing, 10(2), 151-164.

The algorithm mimics the phototaxis behavior of moths toward light sources
(Lévy flights) and the spiral flying path around the flame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_LEVY_BETA = 1.5  # Lévy distribution parameter


class MothSearchAlgorithm(AbstractOptimizer):
    """Moth Search Algorithm.

    Simulates the phototaxis behavior of moths, combining:
    - Lévy flights for global exploration
    - Spiral movement toward light sources for exploitation

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of moths in the population.
        path_finder_ratio: Ratio of moths acting as pathfinders. Default 0.5.


    Example:
        >>> from opt.swarm_intelligence.moth_search import MothSearchAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = MothSearchAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = MothSearchAlgorithm(
        ...     func=shifted_ackley, dim=2,
        ...     lower_bound=-2.768, upper_bound=2.768,
        ...     max_iter=10, seed=42
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
        path_finder_ratio: float = 0.5,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.path_finder_ratio = path_finder_ratio
        self.n_pathfinders = int(population_size * path_finder_ratio)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Moth Search Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize moth population
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Sort by fitness (ascending - minimization)
        sorted_indices = np.argsort(fitness)
        positions = positions[sorted_indices]
        fitness = fitness[sorted_indices]

        # Best solution
        best_solution = positions[0].copy()
        best_fitness = fitness[0]

        for iteration in range(self.max_iter):
            # Update pathfinders using Lévy flight
            for i in range(self.n_pathfinders):
                # Lévy flight
                levy_step = self._levy_flight()
                new_position = positions[i] + levy_step * (positions[i] - best_solution)

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Update followers using spiral movement
            for i in range(self.n_pathfinders, self.population_size):
                # Select a random pathfinder as light source
                light_idx = np.random.randint(self.n_pathfinders)
                light = positions[light_idx]

                # Spiral movement
                distance = np.abs(light - positions[i])
                b = 1.0  # Spiral constant
                t = np.random.uniform(-1, 1)
                new_position = distance * np.exp(b * t) * np.cos(2 * np.pi * t) + light

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    positions[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Re-sort population
            sorted_indices = np.argsort(fitness)
            positions = positions[sorted_indices]
            fitness = fitness[sorted_indices]

        return best_solution, best_fitness

    def _levy_flight(self) -> np.ndarray:
        """Generate Lévy flight step using Mantegna's algorithm.

        Returns:
            Step vector following Lévy distribution.
        """
        import math

        beta = _LEVY_BETA

        # Mantegna's algorithm
        sigma_u = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        sigma_v = 1

        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, sigma_v, self.dim)

        step = u / (np.abs(v) ** (1 / beta))

        return 0.01 * step


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(MothSearchAlgorithm)
