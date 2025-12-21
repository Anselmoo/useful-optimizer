"""Artificial Gorilla Troops Optimizer (GTO).

This module implements the Artificial Gorilla Troops Optimizer,
a metaheuristic algorithm inspired by the social intelligence
of gorilla troops in nature.

Reference:
    Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021).
    Artificial gorilla troops optimizer: A new nature-inspired metaheuristic
    algorithm for global optimization problems.
    International Journal of Intelligent Systems, 36(10), 5887-5958.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for GTO algorithm
_BETA = 3.0  # Coefficient for silverback following
_EXPLORATION_THRESHOLD = 0.5  # Threshold for exploration vs exploitation
_W_MIN = 0.8  # Minimum weight for random walk
_W_MAX = 1.0  # Maximum weight for random walk


class ArtificialGorillaTroopsOptimizer(AbstractOptimizer):
    """Artificial Gorilla Troops Optimizer implementation.

    GTO is inspired by the social behavior of gorillas, including:
    - Exploration: Gorillas move to unknown regions
    - Exploitation: Following the silverback (best solution)
    - Social interactions within the troop

    Attributes:
        func: The objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        population_size: Number of gorillas in the troop.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.artificial_gorilla_troops import (
        ...     ArtificialGorillaTroopsOptimizer,
        ... )
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ArtificialGorillaTroopsOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ArtificialGorillaTroopsOptimizer(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10,
        ...     seed=42,
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
        population_size: int = 50,
        max_iter: int = 500,
    ) -> None:
        """Initialize the GTO optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound for all dimensions.
            upper_bound: Upper bound for all dimensions.
            dim: Number of dimensions.
            population_size: Number of gorillas.
            max_iter: Maximum iterations.
        """
        super().__init__(func, lower_bound, upper_bound, dim)
        self.population_size = population_size
        self.max_iter = max_iter

    def _levy_flight(self, dim: int) -> np.ndarray:
        """Generate Lévy flight step.

        Args:
            dim: Number of dimensions.

        Returns:
            Lévy flight step vector.
        """
        beta = 1.5
        sigma_u = (
            math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)
        sigma_v = 1.0

        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim) * sigma_v

        return u / (np.abs(v) ** (1 / beta))

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the optimization algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Initialize silverback (best solution)
        best_idx = np.argmin(fitness)
        silverback = population[best_idx].copy()
        silverback_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Update parameters
            a = (np.cos(2 * np.random.rand()) + 1) * (
                1 - (iteration + 1) / self.max_iter
            )
            c = a * (2 * np.random.rand() - 1)

            for i in range(self.population_size):
                # Calculate weight
                w = _W_MIN + (_W_MAX - _W_MIN) * np.random.rand()

                if np.random.rand() < _EXPLORATION_THRESHOLD:
                    # Exploration phase
                    if np.abs(c) >= 1:
                        # Move to unknown location (random gorilla)
                        rand_idx = np.random.randint(self.population_size)
                        gr = population[rand_idx]
                        new_position = (
                            self.upper_bound - self.lower_bound
                        ) * np.random.rand(self.dim) + self.lower_bound
                        new_position = w * new_position + (1 - w) * gr
                    else:
                        # Group following
                        r1 = np.random.rand(self.dim)
                        z = np.random.uniform(-c, c, self.dim)
                        h = z * population[i]
                        new_position = (
                            r1 * (np.mean(population, axis=0) - population[i]) + h
                        )
                else:
                    # Exploitation phase - follow silverback
                    r2 = np.random.rand()
                    if r2 >= _EXPLORATION_THRESHOLD:
                        # Follow silverback with Lévy flight
                        levy = self._levy_flight(self.dim)
                        new_position = (
                            silverback
                            - levy * (silverback - population[i])
                            + np.random.randn(self.dim)
                            * (_BETA * (silverback - population[i]))
                        )
                    else:
                        # Young silverbacks compete
                        q = 2 * np.random.rand() - 1
                        new_position = silverback - q * (
                            silverback - population[i]
                        ) * np.random.rand(self.dim)

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update if better
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    # Update silverback if necessary
                    if new_fitness < silverback_fitness:
                        silverback = new_position.copy()
                        silverback_fitness = new_fitness

        return silverback, silverback_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ArtificialGorillaTroopsOptimizer)
