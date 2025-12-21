"""Spotted Hyena Optimizer (SHO) implementation.

This module implements the Spotted Hyena Optimizer, a nature-inspired
metaheuristic algorithm based on the social behavior and hunting
strategies of spotted hyenas.

Reference:
    Dhiman, G., & Kumar, V. (2017). Spotted hyena optimizer: A novel
    bio-inspired based metaheuristic technique for engineering applications.
    Advances in Engineering Software, 114, 48-70.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Algorithm constants
_H_PARAM = 5  # Parameter controlling cluster size
_B_VALUE = 1  # Coefficient for position update


class SpottedHyenaOptimizer(AbstractOptimizer):
    """Spotted Hyena Optimizer.

    The SHO mimics the hunting behavior of spotted hyenas which includes:
    - Searching and tracking prey
    - Encircling prey
    - Attacking prey

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of the search space.
        upper_bound: Upper bound of the search space.
        dim: Dimensionality of the problem.
        max_iter: Maximum number of iterations.
        population_size: Number of hyenas (solutions).


    Example:
        >>> from opt.swarm_intelligence.spotted_hyena import SpottedHyenaOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SpottedHyenaOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SpottedHyenaOptimizer(
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
        """Initialize the Spotted Hyena Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of the search space.
            upper_bound: Upper bound of the search space.
            dim: Dimensionality of the problem.
            max_iter: Maximum number of iterations.
            population_size: Number of hyenas (solutions).
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Spotted Hyena Optimizer.

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
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Main loop
        for iteration in range(self.max_iter):
            # Update h parameter (decreases linearly from 5 to 0)
            h = _H_PARAM - iteration * (_H_PARAM / self.max_iter)

            for i in range(self.population_size):
                # Calculate encircling behavior vectors
                # B is a random vector in [0, 1]
                # E is the distance vector
                b_vec = np.random.rand(self.dim)
                e_vec = np.abs(2 * b_vec * best_solution - population[i])

                # Calculate position update parameters
                a_val = 2 * h * np.random.rand() - h  # a decreases from h to -h
                c_h = np.zeros(self.dim)

                # Form a cluster of hyenas around the prey
                n_cluster = int(np.ceil(np.random.rand() * _H_PARAM))
                n_cluster = min(n_cluster, self.population_size)

                # Select n_cluster best hyenas
                sorted_indices = np.argsort(fitness)[:n_cluster]

                # Calculate cluster center
                for j in sorted_indices:
                    b_j = np.random.rand(self.dim)
                    e_j = np.abs(2 * b_j * best_solution - population[j])
                    c_h += best_solution - a_val * e_j

                c_h /= n_cluster

                # Update position
                new_position = (c_h + best_solution) / 2

                # Add exploration component
                if np.random.rand() > 0.5:
                    new_position += _B_VALUE * e_vec * np.random.rand(self.dim)
                else:
                    new_position -= _B_VALUE * e_vec * np.random.rand(self.dim)

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new solution
                new_fitness = self.func(new_position)

                # Update if better
                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_solution = new_position.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(SpottedHyenaOptimizer)
