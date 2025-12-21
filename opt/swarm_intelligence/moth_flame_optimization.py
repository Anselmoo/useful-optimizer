"""Moth-Flame Optimization (MFO) Algorithm.

This module implements the Moth-Flame Optimization algorithm, a nature-inspired
metaheuristic based on the navigation behavior of moths in nature.

Moths use a mechanism called transverse orientation for navigation. They maintain
a fixed angle with respect to the moon (a distant light source). However, when moths
encounter artificial lights, this mechanism leads to spiral flight paths around flames.

Reference:
    Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired
    heuristic paradigm. Knowledge-Based Systems, 89, 228-249.
    DOI: 10.1016/j.knosys.2015.07.006

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = MothFlameOptimizer(
    ...     func=shifted_ackley,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=10,
    ...     population_size=30,
    ...     max_iter=500,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
    >>> print(f"Best fitness: {best_fitness}")

Attributes:
    func (Callable): The objective function to minimize.
    lower_bound (float): Lower bound of the search space.
    upper_bound (float): Upper bound of the search space.
    dim (int): Dimensionality of the search space.
    population_size (int): Number of moths in the population.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class MothFlameOptimizer(AbstractOptimizer):
    """Moth-Flame Optimization Algorithm.

    This optimizer mimics the navigation behavior of moths around flames:
    - Moths represent candidate solutions
    - Flames represent the best solutions found so far
    - Moths spiral around flames using logarithmic spiral movement
    - Number of flames decreases over iterations for convergence

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of moths.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
        b (float): Logarithmic spiral shape constant.


    Example:
        >>> from opt.swarm_intelligence.moth_flame_optimization import MothFlameOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = MothFlameOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = MothFlameOptimizer(
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
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
        b: float = 1.0,
    ) -> None:
        """Initialize the Moth-Flame Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of moths.
            b: Logarithmic spiral shape constant (default 1.0).
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )
        self.b = b

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Moth-Flame Optimization algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize moth population
        moths = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        moth_fitness = np.array([self.func(moth) for moth in moths])

        # Sort moths by fitness and initialize flames
        sorted_indices = np.argsort(moth_fitness)
        flames = moths[sorted_indices].copy()
        flame_fitness = moth_fitness[sorted_indices].copy()

        # Track best solution
        best_solution = flames[0].copy()
        best_fitness = flame_fitness[0]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Number of flames decreases over iterations
            flame_count = round(
                self.population_size
                - iteration * ((self.population_size - 1) / self.max_iter)
            )

            # Parameter a decreases linearly from -1 to -2
            a = -1 + iteration * ((-1) / self.max_iter)

            for i in range(self.population_size):
                # Select flame index (moths spiral around their corresponding flame)
                flame_idx = min(i, flame_count - 1)

                # Distance to flame
                distance = abs(flames[flame_idx] - moths[i])

                # Random parameter t in [a, 1]
                t = (a - 1) * rng.random(self.dim) + 1

                # Logarithmic spiral movement
                moths[i] = (
                    distance * np.exp(self.b * t) * np.cos(2 * np.pi * t)
                    + flames[flame_idx]
                )

                # Ensure bounds
                moths[i] = np.clip(moths[i], self.lower_bound, self.upper_bound)

            # Update moth fitness
            moth_fitness = np.array([self.func(moth) for moth in moths])

            # Merge moths and flames, then sort to get best solutions
            combined_population = np.vstack([moths, flames[:flame_count]])
            combined_fitness = np.concatenate(
                [moth_fitness, flame_fitness[:flame_count]]
            )

            # Sort and keep best as new flames
            sorted_indices = np.argsort(combined_fitness)
            flames = combined_population[sorted_indices[: self.population_size]].copy()
            flame_fitness = combined_fitness[sorted_indices[: self.population_size]]

            # Update best solution
            if flame_fitness[0] < best_fitness:
                best_solution = flames[0].copy()
                best_fitness = flame_fitness[0]

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(MothFlameOptimizer)
