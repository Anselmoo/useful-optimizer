"""Aquila Optimizer (AO).

This module implements the Aquila Optimizer, a nature-inspired
metaheuristic algorithm based on the hunting behavior of Aquila
(eagle) in nature.

Reference:
    Abualigah, L., Yousri, D., Abd Elaziz, M., Ewees, A. A., Al-qaness, M. A.,
    & Gandomi, A. H. (2021). Aquila optimizer: A novel meta-heuristic
    optimization algorithm.
    Computers & Industrial Engineering, 157, 107250.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for Aquila Optimizer
_ALPHA = 0.1  # Exploitation parameter
_DELTA = 0.1  # Exploitation parameter
_EXPANSION_THRESHOLD_1 = 2 / 3  # First phase transition
_EXPANSION_THRESHOLD_2 = 1 / 3  # Second phase transition


class AquilaOptimizer(AbstractOptimizer):
    """Aquila Optimizer implementation.

    AO is inspired by the hunting strategies of the Aquila eagle:
    1. High soar with vertical stoop (expanded exploration)
    2. Contour flight with short glide attack (narrowed exploration)
    3. Low flight with slow descent attack (expanded exploitation)
    4. Walk and grab prey (narrowed exploitation)

    Attributes:
        func: The objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        population_size: Number of search agents.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.aquila_optimizer import AquilaOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = AquilaOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = AquilaOptimizer(
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
        population_size: int = 50,
        max_iter: int = 500,
    ) -> None:
        """Initialize the Aquila Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound for all dimensions.
            upper_bound: Upper bound for all dimensions.
            dim: Number of dimensions.
            population_size: Number of search agents.
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

    def _quality_function(self, iteration: int, max_iter: int) -> float:
        """Calculate quality function for search behavior.

        Args:
            iteration: Current iteration.
            max_iter: Maximum iterations.

        Returns:
            Quality function value.
        """
        return 2 * np.random.rand() - 1 * (1 - (iteration / max_iter) ** (1 / _ALPHA))

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

        # Initialize best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Calculate mean position
        mean_position = np.mean(population, axis=0)

        for iteration in range(self.max_iter):
            # Update quality function
            qf = self._quality_function(iteration, self.max_iter)

            # Calculate progress ratio (decreases over iterations)
            t_ratio = (self.max_iter - iteration) / self.max_iter

            for i in range(self.population_size):
                rand = np.random.rand()

                if t_ratio > _EXPANSION_THRESHOLD_1:
                    # Phase 1: Expanded exploration (high soar)
                    if rand < _EXPANSION_THRESHOLD_2:
                        # X1: Vertical stoop
                        new_position = (
                            best_solution * (1 - (iteration / self.max_iter))
                            + (mean_position - best_solution) * np.random.rand()
                        )
                    else:
                        # X2: With Lévy flight
                        levy = self._levy_flight(self.dim)
                        d = np.arange(1, self.dim + 1)
                        ub_lb = self.upper_bound - self.lower_bound
                        new_position = (
                            best_solution * levy
                            + population[np.random.randint(self.population_size)]
                            + (ub_lb * np.random.rand() + self.lower_bound)
                            * np.log10(d)
                        )

                elif t_ratio > _EXPANSION_THRESHOLD_2:
                    # Phase 2: Narrowed exploration (contour flight)
                    new_position = (
                        (best_solution - mean_position) * _ALPHA
                        - np.random.rand()
                        + (
                            (self.upper_bound - self.lower_bound) * np.random.rand()
                            + self.lower_bound
                        )
                        * _DELTA
                    )

                elif rand < _EXPANSION_THRESHOLD_2:
                    # Phase 3: Expanded exploitation (low flight)
                    qf_term = (
                        qf * best_solution
                        - ((iteration * 2 / self.max_iter) ** 2) * population[i]
                    )
                    new_position = qf_term + np.random.rand(self.dim) * (
                        best_solution - population[i]
                    )

                else:
                    # Phase 4: Narrowed exploitation (walk and grab)
                    d1 = np.random.uniform(1, self.dim)
                    d2 = np.random.uniform(1, self.dim)
                    levy = self._levy_flight(self.dim)
                    new_position = (
                        best_solution - (d1 * best_solution - d2 * mean_position) * levy
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update if better
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    # Update best if necessary
                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

            # Update mean position
            mean_position = np.mean(population, axis=0)

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(AquilaOptimizer)
