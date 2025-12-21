"""Golden Eagle Optimizer (GEO) implementation.

This module implements the Golden Eagle Optimizer, a nature-inspired
metaheuristic based on the intelligent hunting behavior of golden eagles.

Reference:
    Mohammadi-Balani, A., Nayeri, M. D., Azar, A., & Taghizadeh-Yazdi, M.
    (2021). Golden eagle optimizer: A nature-inspired metaheuristic algorithm.
    Computers & Industrial Engineering, 152, 107050.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_PA_MIN = 0.5  # Minimum propensity to attack
_PA_MAX = 2.0  # Maximum propensity to attack
_PC_MIN = 0.5  # Minimum propensity to cruise
_PC_MAX = 2.0  # Maximum propensity to cruise


class GoldenEagleOptimizer(AbstractOptimizer):
    """Golden Eagle Optimizer.

    The GEO mimics golden eagle hunting strategies:
    - Cruising to explore search space
    - Attacking to exploit prey locations
    - Balance between exploration and exploitation

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of eagles (solutions).


    Example:
        >>> from opt.swarm_intelligence.golden_eagle import GoldenEagleOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = GoldenEagleOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = GoldenEagleOptimizer(
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
        max_iter: int,
        population_size: int = 30,
    ) -> None:
        """Initialize the Golden Eagle Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of eagles (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Golden Eagle Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best solution (prey)
        best_idx = np.argmin(fitness)
        prey = population[best_idx].copy()
        prey_fitness = fitness[best_idx]

        # Store attack vectors
        attack_vectors = np.zeros((self.population_size, self.dim))

        # Main loop
        for iteration in range(self.max_iter):
            # Update propensity parameters
            t_ratio = iteration / self.max_iter

            # Propensity to attack (increases over time)
            pa = _PA_MIN + (_PA_MAX - _PA_MIN) * t_ratio

            # Propensity to cruise (decreases over time)
            pc = _PC_MAX - (_PC_MAX - _PC_MIN) * t_ratio

            for i in range(self.population_size):
                # Random prey selection (occasionally use non-best)
                if np.random.rand() < 0.5:
                    selected_prey = prey
                else:
                    rand_idx = np.random.randint(self.population_size)
                    selected_prey = population[rand_idx]

                # Calculate attack vector
                r1 = np.random.rand()
                r2 = np.random.rand()

                # Cruise component (exploration)
                cruise_vector = (
                    np.random.randn(self.dim)
                    * (self.upper_bound - self.lower_bound)
                    * (1 - t_ratio)
                )

                # Attack component (exploitation)
                attack_vector = pa * r1 * (selected_prey - population[i])

                # Combined movement
                delta_x = pc * r2 * cruise_vector + attack_vector

                # Update position
                new_position = population[i] + delta_x

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness
                    attack_vectors[i] = delta_x

                    if new_fitness < prey_fitness:
                        prey = new_position.copy()
                        prey_fitness = new_fitness

        return prey, prey_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(GoldenEagleOptimizer)
