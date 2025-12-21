"""Honey Badger Algorithm.

Implementation based on:
Hashim, F.A., Houssein, E.H., Hussain, K., Mabrouk, M.S. & Al-Atabany, W. (2022).
Honey Badger Algorithm: New metaheuristic algorithm for solving optimization
problems.
Mathematics and Computers in Simulation, 192, 84-110.

The algorithm mimics the foraging behavior of honey badgers, known for their
intelligence, persistence, and fearlessness in hunting prey and raiding beehives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_BETA = 6.0  # Ability of honey badger to get food (density factor)


class HoneyBadgerAlgorithm(AbstractOptimizer):
    """Honey Badger Algorithm.

    Simulates the intelligent foraging behavior of honey badgers, combining:
    - Digging behavior: Exploitative search near the best solution
    - Honey behavior: Explorative search toward food sources

    Args:
        func: Objective function to minimize.
        lower_bound: Lower bound for the search space.
        upper_bound: Upper bound for the search space.
        dim: Dimensionality of the search space.
        max_iter: Maximum number of iterations.
        population_size: Number of honey badgers.
        beta: Density factor controlling convergence. Default 6.0.


    Example:
        >>> from opt.swarm_intelligence.honey_badger import HoneyBadgerAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = HoneyBadgerAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = HoneyBadgerAlgorithm(
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
        beta: float = _BETA,
    ) -> None:
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.beta = beta

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Honey Badger Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        positions = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate fitness
        fitness = np.array([self.func(pos) for pos in positions])

        # Find prey (best solution)
        prey_idx = np.argmin(fitness)
        prey = positions[prey_idx].copy()
        prey_fitness = fitness[prey_idx]

        for iteration in range(self.max_iter):
            # Decrease intensity factor over iterations
            alpha = self._calculate_alpha(iteration)

            for i in range(self.population_size):
                # Compute smell intensity
                intensity = self._compute_intensity(positions[i], prey)

                # Random flag for search behavior
                flag = np.random.choice([-1, 1])

                # Distance from prey
                distance = prey - positions[i]

                r = np.random.rand()

                if r < 0.5:
                    # Digging phase (exploitation)
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    r5 = np.random.rand()

                    positions[i] = (
                        prey
                        + flag * self.beta * intensity * prey
                        + flag
                        * r3
                        * alpha
                        * distance
                        * np.abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
                    )
                else:
                    # Honey phase (exploration)
                    r6 = np.random.rand()
                    r7 = np.random.rand()

                    positions[i] = (
                        prey
                        + flag * r6 * alpha * distance
                        + r7 * np.random.randn(self.dim)
                    )

                # Boundary handling
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = self.func(positions[i])
                fitness[i] = new_fitness

                # Update prey if better solution found
                if new_fitness < prey_fitness:
                    prey = positions[i].copy()
                    prey_fitness = new_fitness

        return prey, prey_fitness

    def _calculate_alpha(self, iteration: int) -> float:
        """Calculate alpha parameter that decreases over iterations.

        Args:
            iteration: Current iteration number.

        Returns:
            Alpha value controlling search intensity.
        """
        c = 2.0  # Constant
        return c * np.exp(-iteration / self.max_iter)

    def _compute_intensity(self, position: np.ndarray, prey: np.ndarray) -> float:
        """Compute smell intensity based on distance from prey.

        Args:
            position: Current position of honey badger.
            prey: Position of prey (best solution).

        Returns:
            Smell intensity value.
        """
        r2 = np.random.rand()
        distance = np.linalg.norm(position - prey)
        return r2 * (4 * distance**2) / (4 * np.pi * distance**2 + 1e-10)


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(HoneyBadgerAlgorithm)
