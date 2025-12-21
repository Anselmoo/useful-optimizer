"""Osprey Optimization Algorithm (OOA).

This module implements the Osprey Optimization Algorithm, a nature-inspired
metaheuristic algorithm that mimics the hunting behavior of ospreys.

Ospreys are fish-eating birds of prey known for their remarkable hunting
skills. The algorithm simulates their hunting phases: position identification,
fish detection, and attack.

Reference:
    Dehghani, M., Trojovský, P., & Hubálovský, Š. (2023).
    Osprey optimization algorithm: A new bio-inspired metaheuristic algorithm
    for solving engineering optimization problems.
    Frontiers in Mechanical Engineering, 8, 1126450.
    DOI: 10.3389/fmech.2022.1126450

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = OspreyOptimizer(
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


class OspreyOptimizer(AbstractOptimizer):
    """Osprey Optimization Algorithm optimizer.

    This algorithm simulates the hunting behavior of ospreys, including:
    1. Position identification phase - ospreys identify fish positions
    2. Carrying fish to suitable position - moving toward best positions
    3. Attack phase - exploitation around promising solutions

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        population_size: Number of ospreys in the population.
        max_iter: Maximum number of iterations.


    Example:
        >>> from opt.swarm_intelligence.osprey_optimizer import OspreyOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = OspreyOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = OspreyOptimizer(
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
        population_size: int = 30,
        max_iter: int = 100,
    ) -> None:
        """Initialize Osprey Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            population_size: Number of ospreys. Defaults to 30.
            max_iter: Maximum iterations. Defaults to 100.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.population_size = population_size

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Osprey Optimization Algorithm.

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
            for i in range(self.population_size):
                # Phase 1: Position identification (exploration)
                # Select random fish position
                fish_idx = np.random.randint(self.population_size)
                while fish_idx == i:
                    fish_idx = np.random.randint(self.population_size)

                fish_position = population[fish_idx]

                # Calculate new position based on fish location
                r1 = np.random.random(self.dim)
                r2 = np.random.random()

                if fitness[fish_idx] < fitness[i]:
                    # Move toward better fish position
                    new_position = population[i] + r1 * (
                        fish_position - population[i] * (1 + r2)
                    )
                else:
                    # Move away from worse position
                    new_position = population[i] + r1 * (
                        population[i] - fish_position * (1 + r2)
                    )

                # Boundary handling
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_fitness = self.func(new_position)

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                # Phase 2: Attack phase (exploitation)
                t = 1 - iteration / self.max_iter
                r3 = np.random.random(self.dim)
                r4 = np.random.random()

                # Attack toward best solution
                attack_position = best_solution + (
                    r3 * (best_solution - population[i]) * t
                    + (2 * r4 - 1) * t * (self.upper_bound - self.lower_bound) / 100
                )

                attack_position = np.clip(
                    attack_position, self.lower_bound, self.upper_bound
                )
                attack_fitness = self.func(attack_position)

                if attack_fitness < fitness[i]:
                    population[i] = attack_position
                    fitness[i] = attack_fitness

                # Update best solution
                if fitness[i] < best_fitness:
                    best_solution = population[i].copy()
                    best_fitness = fitness[i]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(OspreyOptimizer)
