"""Grasshopper Optimization Algorithm (GOA).

This module implements the Grasshopper Optimization Algorithm, a nature-inspired
metaheuristic based on the swarming behavior of grasshoppers in nature.

Grasshoppers naturally form swarms and move toward food sources while avoiding
collisions with each other. The algorithm mimics this behavior with social forces
(attraction/repulsion) and movement toward the best solution.

Reference:
    Saremi, S., Mirjalili, S., & Lewis, A. (2017). Grasshopper Optimisation
    Algorithm: Theory and application. Advances in Engineering Software,
    105, 30-47. DOI: 10.1016/j.advengsoft.2017.01.004

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = GrasshopperOptimizer(
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
    population_size (int): Number of grasshoppers in the swarm.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

# Constants for social force function
_ATTRACTION_INTENSITY = 0.5  # f parameter
_ATTRACTIVE_LENGTH_SCALE = 1.5  # l parameter
_C_MAX = 1.0  # Maximum coefficient for social forces
_C_MIN = 0.00001  # Minimum coefficient for social forces
_DISTANCE_EPSILON = 1e-10  # Small value to avoid division by zero


class GrasshopperOptimizer(AbstractOptimizer):
    """Grasshopper Optimization Algorithm.

    This optimizer mimics the swarming behavior of grasshoppers:
    - Grasshoppers attract/repel each other based on distance
    - Social forces decrease over iterations for convergence
    - Movement is guided by the best solution found (target)
    - Parameter c decreases from c_max to c_min for exploration to exploitation

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of grasshoppers.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.
        c_max (float): Maximum social force coefficient.
        c_min (float): Minimum social force coefficient.
        f (float): Attraction intensity.
        l (float): Attractive length scale.


    Example:
        >>> from opt.swarm_intelligence.grasshopper_optimization import GrasshopperOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = GrasshopperOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = GrasshopperOptimizer(
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
        max_iter: int = 1000,
        seed: int | None = None,
        population_size: int = 100,
        c_max: float = _C_MAX,
        c_min: float = _C_MIN,
        f: float = _ATTRACTION_INTENSITY,
        l: float = _ATTRACTIVE_LENGTH_SCALE,
    ) -> None:
        """Initialize the Grasshopper Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of grasshoppers.
            c_max: Maximum social force coefficient.
            c_min: Minimum social force coefficient.
            f: Attraction intensity parameter.
            l: Attractive length scale parameter.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )
        self.c_max = c_max
        self.c_min = c_min
        self.f = f
        self.l = l

    def _social_force(self, distance: float) -> float:
        """Calculate the social force between two grasshoppers.

        The s function models attraction and repulsion:
        s(r) = f * exp(-r/l) - exp(-r)

        Args:
            distance: Distance between two grasshoppers.

        Returns:
            Social force value (positive = attraction, negative = repulsion).
        """
        return self.f * np.exp(-distance / self.l) - np.exp(-distance)

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Grasshopper Optimization Algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize grasshopper population
        grasshoppers = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(gh) for gh in grasshoppers])

        # Find target (best solution)
        best_idx = np.argmin(fitness)
        target = grasshoppers[best_idx].copy()
        target_fitness = fitness[best_idx]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update coefficient c (decreases from c_max to c_min)
            c = self.c_max - iteration * ((self.c_max - self.c_min) / self.max_iter)

            # Calculate normalized bounds for social force scaling
            ub = self.upper_bound
            lb = self.lower_bound

            # Update each grasshopper
            new_positions = np.zeros_like(grasshoppers)

            for i in range(self.population_size):
                social_sum = np.zeros(self.dim)

                for j in range(self.population_size):
                    if i != j:
                        # Calculate distance between grasshoppers
                        dist_vec = grasshoppers[j] - grasshoppers[i]
                        distance = np.linalg.norm(dist_vec)

                        # Avoid division by zero
                        if distance > _DISTANCE_EPSILON:
                            # Normalize distance
                            unit_vec = dist_vec / distance

                            # Normalize distance to [1, 4] as in original paper
                            normalized_dist = 2 + (distance % 2)

                            # Social interaction force
                            s = self._social_force(normalized_dist)

                            # Accumulate social forces
                            social_sum += c * ((ub - lb) / 2) * s * unit_vec

                # Update position: social forces + target attraction
                new_positions[i] = c * social_sum + target

                # Ensure bounds
                new_positions[i] = np.clip(
                    new_positions[i], self.lower_bound, self.upper_bound
                )

            # Update grasshoppers
            grasshoppers = new_positions

            # Update fitness
            fitness = np.array([self.func(gh) for gh in grasshoppers])

            # Update target
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < target_fitness:
                target = grasshoppers[best_idx].copy()
                target_fitness = fitness[best_idx]

        return target, target_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(GrasshopperOptimizer)
