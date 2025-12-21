"""Chimp Optimization Algorithm (ChOA) implementation.

This module implements the Chimp Optimization Algorithm, a swarm-based
metaheuristic inspired by the social intelligence and hunting behavior
of chimpanzees.

Reference:
    Khishe, M., & Mosavi, M. R. (2020). Chimp optimization algorithm.
    Expert Systems with Applications, 149, 113338.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_A_MAX = 2.5  # Maximum value for parameter a
_F_MAX = 2.0  # Maximum chaos factor


class ChimpOptimizationAlgorithm(AbstractOptimizer):
    """Chimp Optimization Algorithm optimizer.

    The ChOA mimics chimpanzee hunting behavior:
    - Social hierarchy (attacker, barrier, chaser, driver)
    - Driving and chasing prey
    - Attacking and surrounding prey

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of chimps (solutions).


    Example:
        >>> from opt.swarm_intelligence.chimp_optimization import ChimpOptimizationAlgorithm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ChimpOptimizationAlgorithm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ChimpOptimizationAlgorithm(
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
        """Initialize the Chimp Optimization Algorithm.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of chimps (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Chimp Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(ind) for ind in population])

        # Sort by fitness to get 4 best chimps
        sorted_indices = np.argsort(fitness)

        # Four best chimps (attacker, barrier, chaser, driver)
        attacker = population[sorted_indices[0]].copy()
        barrier = population[sorted_indices[1]].copy()
        chaser = population[sorted_indices[2]].copy()
        driver = population[sorted_indices[3]].copy()

        best_solution = attacker.copy()
        best_fitness = fitness[sorted_indices[0]]

        # Main loop
        for iteration in range(self.max_iter):
            # Update parameter a (decreases from A_MAX to 0)
            a = _A_MAX - iteration * (_A_MAX / self.max_iter)

            # Chaos factor (decreases from F_MAX to 0)
            f = _F_MAX * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                # Random coefficients
                r1, r2, r3, r4 = np.random.rand(4)
                c1, c2, c3, c4 = np.random.rand(4)

                # Calculate A coefficients for each leader
                a1 = 2 * a * r1 - a
                a2 = 2 * a * r2 - a
                a3 = 2 * a * r3 - a
                a4 = 2 * a * r4 - a

                # Distance from each leader
                d_attacker = np.abs(c1 * attacker - population[i])
                d_barrier = np.abs(c2 * barrier - population[i])
                d_chaser = np.abs(c3 * chaser - population[i])
                d_driver = np.abs(c4 * driver - population[i])

                # Position updates from each leader
                x1 = attacker - a1 * d_attacker
                x2 = barrier - a2 * d_barrier
                x3 = chaser - a3 * d_chaser
                x4 = driver - a4 * d_driver

                # Combined position with chaos
                new_position = (x1 + x2 + x3 + x4) / 4

                # Add chaotic movement
                if np.random.rand() < 0.5:
                    new_position += f * np.random.randn(self.dim)

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate and update
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

            # Update leaders
            sorted_indices = np.argsort(fitness)
            attacker = population[sorted_indices[0]].copy()
            barrier = population[sorted_indices[1]].copy()
            chaser = population[sorted_indices[2]].copy()
            driver = population[sorted_indices[3]].copy()

            if fitness[sorted_indices[0]] < best_fitness:
                best_solution = attacker.copy()
                best_fitness = fitness[sorted_indices[0]]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = ChimpOptimizationAlgorithm(
        func=shifted_ackley,
        lower_bound=-2.768,
        upper_bound=2.768,
        dim=2,
        max_iter=100,
        population_size=30,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
