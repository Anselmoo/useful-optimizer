"""Emperor Penguin Optimizer (EPO) implementation.

This module implements the Emperor Penguin Optimizer, a nature-inspired
metaheuristic based on the huddling behavior of emperor penguins
to survive the harsh Antarctic winter.

Reference:
    Dhiman, G., & Kumar, V. (2018). Emperor penguin optimizer: A bio-inspired
    algorithm for engineering problems. Knowledge-Based Systems, 159, 20-50.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_M = 2  # Movement parameter
_F_INIT = 2  # Initial temperature coefficient
_L_INIT = 1.5  # Initial huddling coefficient


class EmperorPenguinOptimizer(AbstractOptimizer):
    """Emperor Penguin Optimizer.

    The EPO mimics emperor penguin huddling behavior:
    - Penguins move to avoid wind (exploration)
    - Penguins huddle together for warmth (exploitation)
    - Group dynamics help find optimal positions

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of penguins (solutions).


    Example:
        >>> from opt.swarm_intelligence.emperor_penguin import EmperorPenguinOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = EmperorPenguinOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = EmperorPenguinOptimizer(
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
    ) -> None:
        """Initialize the Emperor Penguin Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of penguins (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Emperor Penguin Optimizer.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Find best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Update temperature-related parameters
            t = iteration / self.max_iter

            # Temperature profile (ensure non-negative for sqrt)
            temp = max(0, _F_INIT * np.exp(-t * _M) - t * np.exp(-t))

            # Huddle coefficient
            lam = _L_INIT * np.exp(-t)

            for i in range(self.population_size):
                # Calculate polygon grid accuracy
                p_grid = np.abs(best_solution - population[i])

                # Calculate social forces
                a = np.random.rand()  # Temperature gradient
                c = np.random.rand()  # Collision avoidance

                # Move to avoid wind (exploration)
                s = np.sqrt(temp) * np.random.randn(self.dim)

                # Huddle towards warmer positions (exploitation)
                d = np.abs(lam * best_solution - population[i])

                # Update position
                new_position = best_solution - a * (c * p_grid - s * d)

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(EmperorPenguinOptimizer)
