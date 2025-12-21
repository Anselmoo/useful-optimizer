"""African Vultures Optimization Algorithm (AVOA).

This module implements the African Vultures Optimization Algorithm,
a nature-inspired metaheuristic based on the foraging and navigation
behaviors of African vultures.

Reference:
    Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021).
    African vultures optimization algorithm: A new nature-inspired
    metaheuristic algorithm for global optimization problems.
    Computers & Industrial Engineering, 158, 107408.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for AVOA algorithm
_P1 = 0.6  # Probability for selecting first best vulture
_P2 = 0.4  # Probability for selecting second best vulture
_P3 = 0.6  # Probability for exploration vs exploitation behavior
_OMEGA = 0.4  # Threshold for satiation rate behavior change
_L1 = 0.8  # Lower bound for satiation rate
_L2 = 0.2  # Upper bound decrease rate


class AfricanVulturesOptimizer(AbstractOptimizer):
    """African Vultures Optimization Algorithm implementation.

    AVOA mimics the behavior of African vultures including:
    - Group behavior: Vultures split into groups around best solutions
    - Exploration: Random movement and rotation flight
    - Exploitation: Different attack strategies based on satiation

    Attributes:
        func: The objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        population_size: Number of vultures.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.african_vultures_optimizer import AfricanVulturesOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = AfricanVulturesOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = AfricanVulturesOptimizer(
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
        """Initialize the AVOA optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound for all dimensions.
            upper_bound: Upper bound for all dimensions.
            dim: Number of dimensions.
            population_size: Number of vultures.
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

    def _calculate_satiation(self, iteration: int) -> float:
        """Calculate satiation rate (hunger).

        Args:
            iteration: Current iteration.

        Returns:
            Satiation rate (lower = more hungry).
        """
        z = np.random.uniform(-1, 1)
        h = np.random.uniform(-2, 2)
        t = h * (
            np.sin(np.pi / 2 * iteration / self.max_iter) ** z
            + np.cos(np.pi / 2 * iteration / self.max_iter)
            - 1
        )
        return (2 * np.random.rand() + 1) * z * (1 - iteration / self.max_iter) + t

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

        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        # Best and second best vultures
        best_vulture_1 = population[0].copy()
        best_fitness_1 = fitness[0]
        best_vulture_2 = population[1].copy()
        best_fitness_2 = fitness[1]

        for iteration in range(self.max_iter):
            # Calculate satiation rate
            satiation = self._calculate_satiation(iteration)
            abs_sat = np.abs(satiation)

            for i in range(self.population_size):
                # Select reference vulture
                if np.random.rand() < _P1:
                    reference_vulture = best_vulture_1
                else:
                    reference_vulture = best_vulture_2

                # Random factor
                f = 2 * np.random.rand() * satiation + satiation

                if abs_sat >= 1:
                    # Exploration phase
                    if np.random.rand() < _P3:
                        # Random location selection
                        r1 = np.random.randint(self.population_size)
                        d = np.abs(population[r1] - population[i]) * f
                        new_position = reference_vulture - d
                    else:
                        # Rotation flight
                        s1 = (
                            reference_vulture
                            * np.random.rand(self.dim)
                            * (population[i] / (2 * np.pi))
                        )
                        s2 = reference_vulture * np.cos(population[i])
                        new_position = reference_vulture - (s1 + s2)

                elif abs_sat >= _OMEGA:
                    # Exploitation phase 1
                    if np.random.rand() < _P3:
                        # Siege fight
                        d = np.abs(reference_vulture - population[i])
                        new_position = reference_vulture - f * d
                    else:
                        # Rotation flight in siege
                        s1 = reference_vulture * (
                            np.random.rand(self.dim) * population[i]
                        )
                        s2 = reference_vulture * np.cos(population[i])
                        a1 = (
                            best_vulture_1
                            - (best_vulture_1 * population[i]) / (s1 + 1e-10) * f
                        )
                        a2 = (
                            best_vulture_2
                            - (best_vulture_2 * population[i]) / (s2 + 1e-10) * f
                        )
                        new_position = (a1 + a2) / 2

                # Exploitation phase 2 (aggressive attack)
                elif np.random.rand() < _P3:
                    # Siege fight with Lévy
                    levy = self._levy_flight(self.dim)
                    d = np.abs(reference_vulture - population[i])
                    new_position = reference_vulture - np.abs(d) * f * levy

                else:
                    # Accumulated group attack
                    a1 = best_vulture_1 - ((best_vulture_1 - population[i]) * f)
                    a2 = best_vulture_2 - ((best_vulture_2 - population[i]) * f)
                    new_position = (a1 + a2) / 2

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update if better
                new_fitness = self.func(new_position)
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

            # Update best vultures
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            if fitness[0] < best_fitness_1:
                best_vulture_2 = best_vulture_1.copy()
                best_fitness_2 = best_fitness_1
                best_vulture_1 = population[0].copy()
                best_fitness_1 = fitness[0]
            elif fitness[0] < best_fitness_2:
                best_vulture_2 = population[0].copy()
                best_fitness_2 = fitness[0]

        return best_vulture_1, best_fitness_1


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = AfricanVulturesOptimizer(
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
