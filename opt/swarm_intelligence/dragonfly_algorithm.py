"""Dragonfly Algorithm (DA).

This module implements the Dragonfly Algorithm, a swarm intelligence optimization
algorithm based on the static and dynamic swarming behaviors of dragonflies.

Dragonflies form sub-swarms for hunting (static swarm) and migrate in one direction
(dynamic swarm). These behaviors map to exploration and exploitation in optimization.

Reference:
    Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization
    technique for solving single-objective, discrete, and multi-objective problems.
    Neural Computing and Applications, 27(4), 1053-1073.
    DOI: 10.1007/s00521-015-1920-1

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = DragonflyOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=10,
    ...     population_size=30,
    ...     max_iter=500,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
    >>> print(f"Best fitness: {best_fitness}")

Attributes:
    func (Callable): The objective function to minimize.
    lower_bound (float): Lower bound of the search space.
    upper_bound (float): Upper bound of the search space.
    dim (int): Dimensionality of the search space.
    population_size (int): Number of dragonflies in the swarm.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class DragonflyOptimizer(AbstractOptimizer):
    """Dragonfly Algorithm.

    This optimizer mimics the swarming behavior of dragonflies:
    - Separation: Avoid collision with neighbors
    - Alignment: Match velocity with neighbors
    - Cohesion: Move toward center of neighbors
    - Attraction: Move toward food source
    - Distraction: Move away from enemies

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of dragonflies.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
    ) -> None:
        """Initialize the Dragonfly Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of dragonflies.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )

    def _find_neighbors(
        self, position: np.ndarray, all_positions: np.ndarray, radius: float
    ) -> np.ndarray:
        """Find neighbors within radius.

        Args:
            position: Current dragonfly position.
            all_positions: All dragonfly positions.
            radius: Neighborhood radius.

        Returns:
            Array of neighbor positions.
        """
        distances = np.linalg.norm(all_positions - position, axis=1)
        neighbor_mask = (distances < radius) & (distances > 0)
        return all_positions[neighbor_mask]

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Dragonfly Algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize dragonfly population and velocities
        dragonflies = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.zeros((self.population_size, self.dim))

        # Evaluate initial fitness
        fitness = np.array([self.func(df) for df in dragonflies])

        # Track best (food) and worst (enemy) solutions
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        food = dragonflies[best_idx].copy()
        food_fitness = fitness[best_idx]
        enemy = dragonflies[worst_idx].copy()

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update weights (decrease exploration, increase exploitation)
            w = 0.9 - iteration * ((0.9 - 0.4) / self.max_iter)
            # Update radius (decreases over iterations)
            radius = (self.upper_bound - self.lower_bound) * 0.1 + (
                (self.upper_bound - self.lower_bound)
                * (self.max_iter - iteration)
                / self.max_iter
            )

            # Adaptive parameters (increase over iterations)
            s = 2 * rng.random() * (iteration / self.max_iter)  # separation
            a = 2 * rng.random() * (iteration / self.max_iter)  # alignment
            c = 2 * rng.random() * (iteration / self.max_iter)  # cohesion
            f = 2 * rng.random()  # food attraction
            e = rng.random() * (1 - iteration / self.max_iter)  # enemy distraction

            for i in range(self.population_size):
                neighbors = self._find_neighbors(dragonflies[i], dragonflies, radius)

                if len(neighbors) > 0:
                    # Separation: avoid neighbors
                    separation = np.sum(neighbors - dragonflies[i], axis=0)

                    # Alignment: match velocity with neighbors
                    alignment = np.mean(velocities, axis=0)

                    # Cohesion: move toward neighbor center
                    cohesion = np.mean(neighbors, axis=0) - dragonflies[i]

                    # Food attraction
                    food_attraction = food - dragonflies[i]

                    # Enemy distraction
                    enemy_distraction = enemy + dragonflies[i]

                    # Update velocity
                    velocities[i] = (
                        w * velocities[i]
                        + s * separation
                        + a * alignment
                        + c * cohesion
                        + f * food_attraction
                        + e * enemy_distraction
                    )

                    # Update position
                    dragonflies[i] = dragonflies[i] + velocities[i]
                else:
                    # Levy flight for isolated dragonflies
                    levy = self._levy_flight(rng)
                    dragonflies[i] = dragonflies[i] + levy * dragonflies[i]

                # Ensure bounds
                dragonflies[i] = np.clip(
                    dragonflies[i], self.lower_bound, self.upper_bound
                )

            # Update fitness
            fitness = np.array([self.func(df) for df in dragonflies])

            # Update food (best) and enemy (worst)
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)

            if fitness[best_idx] < food_fitness:
                food = dragonflies[best_idx].copy()
                food_fitness = fitness[best_idx]

            enemy = dragonflies[worst_idx].copy()

        return food, food_fitness

    def _levy_flight(self, rng: np.random.Generator) -> np.ndarray:
        """Generate Levy flight step.

        Args:
            rng: Random number generator.

        Returns:
            Levy flight step vector.
        """
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = rng.normal(0, sigma, self.dim)
        v = rng.normal(0, 1, self.dim)

        return u / (np.abs(v) ** (1 / beta))


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(DragonflyOptimizer)
