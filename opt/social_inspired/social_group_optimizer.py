"""Social Group Optimization Algorithm.

This module implements the Social Group Optimization (SGO) algorithm,
a social-inspired metaheuristic based on human social behavior.

The algorithm simulates social interaction behaviors including improving,
acquiring knowledge from others, and self-introspection.

Reference:
    Satapathy, S. C., & Naik, A. (2016).
    Social group optimization (SGO): A new population evolutionary optimization
    technique.
    Complex & Intelligent Systems, 2(3), 173-203.
    DOI: 10.1007/s40747-016-0022-8

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = SocialGroupOptimizer(
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


class SocialGroupOptimizer(AbstractOptimizer):
    """Social Group Optimization algorithm.

    This algorithm simulates human social behaviors:
    1. Improving phase - learning from the best person
    2. Acquiring phase - learning from other group members
    3. Self-introspection - exploring individually

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of individuals in the social group.
        max_iter: Maximum number of iterations.
        c: Self-introspection coefficient.


    Example:
        >>> from opt.social_inspired.social_group_optimizer import SocialGroupOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SocialGroupOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SocialGroupOptimizer(
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
        population_size: int = 30,
        max_iter: int = 100,
        c: float = 0.2,
    ) -> None:
        """Initialize Social Group Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of individuals. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
            c: Self-introspection coefficient. Defaults to 0.2.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size
        self.c = c

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Social Group Optimization algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize population (social group)
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Update self-introspection coefficient
            c_current = self.c * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                new_position = population[i].copy()

                # Phase 1: Improving phase (learn from best)
                r1 = np.random.random(self.dim)
                improving_component = r1 * (best_solution - population[i])

                # Phase 2: Acquiring phase (learn from random member)
                j = np.random.randint(self.population_size)
                while j == i:
                    j = np.random.randint(self.population_size)

                r2 = np.random.random(self.dim)
                if fitness[j] < fitness[i]:
                    acquiring_component = r2 * (population[j] - population[i])
                else:
                    acquiring_component = r2 * (population[i] - population[j])

                # Phase 3: Self-introspection (individual exploration)
                r3 = np.random.uniform(-1, 1, self.dim)
                introspection_component = (
                    c_current * r3 * (self.upper_bound - self.lower_bound)
                )

                # Combine all phases
                new_position = (
                    population[i]
                    + improving_component
                    + acquiring_component
                    + introspection_component
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


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = SocialGroupOptimizer(
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
