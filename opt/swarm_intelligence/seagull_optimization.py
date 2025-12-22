"""Seagull Optimization Algorithm (SOA) implementation.

This module implements the Seagull Optimization Algorithm, a bio-inspired
metaheuristic based on the migration and attack behavior of seagulls.

Reference:
    Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: Theory
    and its applications for large-scale industrial engineering problems.
    Knowledge-Based Systems, 165, 169-196.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_A_MAX = 2.0  # Maximum value for coefficient A
_FC = 2.0  # Frequency control parameter
_U_PARAM = 1.0  # Spiral shape parameter
_V_PARAM = 1.0  # Spiral shape parameter


class SeagullOptimizationAlgorithm(AbstractOptimizer):
    """Seagull Optimization Algorithm optimizer.

    The SOA mimics seagull behavior:
    - Migration behavior (collision avoidance and moving toward best)
    - Attacking behavior (spiral movement)

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of seagulls (solutions).


    Example:
        >>> from opt.swarm_intelligence.seagull_optimization import SeagullOptimizationAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SeagullOptimizationAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SeagullOptimizationAlgorithm(
        ...     func=shifted_ackley, dim=2, lower_bound=-2.768, upper_bound=2.768, max_iter=10
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
        """Initialize the Seagull Optimization Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of seagulls (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Seagull Optimization Algorithm.

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
            # Update A (decreases linearly from A_MAX to 0)
            a_coef = _A_MAX - iteration * (_A_MAX / self.max_iter)

            for i in range(self.population_size):
                # Random parameters
                b_coef = 2 * a_coef**2 * np.random.rand()

                # Migration behavior
                # Collision avoidance
                cs = a_coef * population[i]

                # Movement toward best solution
                ms = b_coef * (best_solution - population[i])

                # New position after migration
                ds = cs + ms

                # Attacking behavior (spiral movement)
                r = np.random.rand()
                theta = 2 * np.pi * r

                # Spiral coordinates
                x_spiral = r * np.cos(theta)
                y_spiral = r * np.sin(theta)
                z_spiral = r * theta

                # Final position (combine spiral with direction)
                new_position = ds * x_spiral * y_spiral * z_spiral + best_solution

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

    run_demo(SeagullOptimizationAlgorithm)
