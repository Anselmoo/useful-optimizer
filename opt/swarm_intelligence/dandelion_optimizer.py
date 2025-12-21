"""Dandelion Optimizer (DO).

This module implements the Dandelion Optimizer, a bio-inspired metaheuristic
algorithm based on the seed dispersal behavior of dandelions.

Dandelions disperse seeds through wind, with seeds traveling in different
patterns depending on wind conditions - from gentle floating to long-distance
travel.

Reference:
    Zhao, S., Zhang, T., Ma, S., & Chen, M. (2022).
    Dandelion Optimizer: A nature-inspired metaheuristic algorithm for
    engineering applications.
    Engineering Applications of Artificial Intelligence, 114, 105075.
    DOI: 10.1016/j.engappai.2022.105075

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = DandelionOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     population_size=30,
    ...     max_iter=100,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class DandelionOptimizer(AbstractOptimizer):
    """Dandelion Optimizer algorithm.

    This algorithm simulates dandelion seed dispersal:
    1. Rising stage - seeds float upward with updraft
    2. Descending stage - seeds fall under gravity
    3. Landing stage - seeds settle and germinate

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of dandelion seeds.
        max_iter: Maximum number of iterations.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int = 30,
        max_iter: int = 100,
    ) -> None:
        """Initialize Dandelion Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of seeds. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Dandelion Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            t = iteration / self.max_iter

            for i in range(self.population_size):
                r = np.random.random()

                if r < 0.3:
                    # Rising stage (exploration)
                    # Seeds rise with random wind patterns
                    alpha = np.random.random()
                    theta = 2 * np.pi * np.random.random()

                    # Logarithmic spiral movement
                    x = np.exp(alpha * theta) * np.cos(theta)
                    y = np.exp(alpha * theta) * np.sin(theta)

                    # Random perturbation with wind
                    wind = np.random.standard_normal(self.dim) * (1 - t)
                    new_position = (
                        population[i]
                        + x * (best_solution - population[i])
                        + y * wind * (self.upper_bound - self.lower_bound) * 0.1
                    )

                elif r < 0.7:
                    # Descending stage (transition)
                    # Seeds fall in random directions
                    mean_pos = np.mean(population, axis=0)
                    r1 = np.random.random(self.dim)
                    r2 = np.random.random()

                    # Move toward mean position with randomness
                    new_position = (
                        population[i]
                        + r1 * (mean_pos - population[i])
                        + r2
                        * np.random.standard_normal(self.dim)
                        * (1 - t)
                        * (self.upper_bound - self.lower_bound)
                        * 0.05
                    )

                else:
                    # Landing stage (exploitation)
                    # Seeds settle near best position
                    levy_step = self._levy_flight()
                    delta = (1 - t) ** 2

                    new_position = best_solution + levy_step * delta * (
                        population[i] - best_solution
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_fitness = self.func(new_position)

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness

    def _levy_flight(self) -> np.ndarray:
        """Generate Levy flight step.

        Returns:
            Levy flight step vector.
        """
        from scipy.special import gamma

        beta = 1.5
        sigma = (
            gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        u = np.random.standard_normal(self.dim) * sigma
        v = np.random.standard_normal(self.dim)

        step = u / (np.abs(v) ** (1 / beta))
        return step * 0.01


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = DandelionOptimizer(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        population_size=30,
        max_iter=100,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
