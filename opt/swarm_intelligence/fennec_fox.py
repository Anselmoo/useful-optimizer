"""Fennec Fox Optimization (FFO) Algorithm.

This module implements the Fennec Fox Optimization algorithm, a nature-inspired
metaheuristic based on the survival behaviors of fennec foxes in the desert.

Fennec foxes use two main strategies: seeking prey and escaping from predators.
Their large ears help them detect prey underground and predators from afar.

Reference:
    Trojovská, E., Dehghani, M., & Trojovský, P. (2023).
    Fennec Fox Optimization: A New Nature-Inspired Optimization Algorithm.
    IEEE Access, 10, 84417-84443.
    DOI: 10.1109/ACCESS.2022.3197745

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = FennecFoxOptimizer(
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


class FennecFoxOptimizer(AbstractOptimizer):
    """Fennec Fox Optimization algorithm optimizer.

    This algorithm simulates fennec fox behaviors:
    1. Prey seeking phase - exploration by searching for food
    2. Escape from predators - exploitation by moving to safe areas

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of foxes in the population.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.fennec_fox import FennecFoxOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = FennecFoxOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = FennecFoxOptimizer(
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
    ) -> None:
        """Initialize Fennec Fox Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of foxes. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Fennec Fox Optimization Algorithm.

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
                r1 = np.random.random()

                if r1 < 0.5:
                    # Phase 1: Prey seeking (exploration)
                    # Select random prey position
                    prey_idx = np.random.randint(self.population_size)
                    prey = population[prey_idx]

                    # Calculate new position
                    r2 = np.random.random(self.dim)
                    r3 = np.random.random()

                    new_position = population[i] + r2 * (prey - r3 * population[i])

                else:
                    # Phase 2: Escape from predators (exploitation)
                    # Move toward best position (safe area)
                    r4 = np.random.random(self.dim)
                    r5 = np.random.random()

                    # Escape coefficient decreases over time
                    escape_factor = (1 - t) * (2 * r5 - 1)

                    new_position = best_solution + escape_factor * r4 * (
                        best_solution - population[i]
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

            # Additional local search around best solution
            r6 = np.random.random(self.dim)
            local_search = best_solution + (2 * r6 - 1) * (1 - t) * 0.1 * (
                self.upper_bound - self.lower_bound
            )
            local_search = np.clip(local_search, self.lower_bound, self.upper_bound)
            local_fitness = self.func(local_search)

            if local_fitness < best_fitness:
                best_solution = local_search.copy()
                best_fitness = local_fitness

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import shifted_ackley

    optimizer = FennecFoxOptimizer(
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
