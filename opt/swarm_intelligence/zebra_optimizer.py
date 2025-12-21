"""Zebra Optimization Algorithm (ZOA).

This module implements the Zebra Optimization Algorithm, a nature-inspired
metaheuristic based on the foraging and defense behaviors of zebras.

Zebras exhibit two main behaviors: foraging (searching for food and water)
and defense against predators through collective movement and vigilance.

Reference:
    Trojovská, E., Dehghani, M., & Trojovský, P. (2022).
    Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm
    for Solving Optimization Problems.
    IEEE Access, 10, 49445-49473.
    DOI: 10.1109/ACCESS.2022.3172789

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = ZebraOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-2.768,
    ...     upper_bound=2.768,
    ...     dim=2,
    ...     population_size=30,
    ...     max_iter=100
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class ZebraOptimizer(AbstractOptimizer):
    """Zebra Optimization Algorithm optimizer.

    This algorithm simulates zebra herd behaviors:
    1. Foraging phase - searching for food sources (exploration)
    2. Defense phase - escaping from predators (exploitation)

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of zebras in the herd.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.zebra_optimizer import ZebraOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ZebraOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ZebraOptimizer(
        ...     func=shifted_ackley,
        ...     dim=2,
        ...     lower_bound=-2.768,
        ...     upper_bound=2.768,
        ...     max_iter=10
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
        """Initialize Zebra Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of zebras. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Zebra Optimization Algorithm.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Initialize herd
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([self.func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Probability of foraging behavior
            p_s = 0.5 * (1 - iteration / self.max_iter)

            for i in range(self.population_size):
                r = np.random.random()

                if r < p_s:
                    # Phase 1: Foraging behavior (exploration)
                    # Zebras search for food/water sources
                    i_food = np.random.randint(self.population_size)
                    while i_food == i:
                        i_food = np.random.randint(self.population_size)

                    food_source = population[i_food]

                    r1 = np.random.random(self.dim)
                    r2 = np.random.random()

                    if fitness[i_food] < fitness[i]:
                        # Move toward better food source
                        new_position = population[i] + r1 * (
                            food_source - r2 * population[i]
                        )
                    else:
                        # Move away from worse position
                        new_position = population[i] + r1 * (
                            population[i] - food_source
                        )

                else:
                    # Phase 2: Defense from predators (exploitation)
                    # Zebras escape toward the herd leader (best solution)
                    attack_power = 0.01 * (1 - (iteration / self.max_iter) ** 2)
                    r3 = np.random.random(self.dim)
                    r4 = np.random.random()

                    # Escape toward best position with decreasing randomness
                    new_position = best_solution + r3 * (
                        best_solution - attack_power * r4 * population[i]
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
    from opt.demo import run_demo

    run_demo(ZebraOptimizer)
