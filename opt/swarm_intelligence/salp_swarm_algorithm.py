"""Salp Swarm Algorithm (SSA).

This module implements the Salp Swarm Algorithm, a nature-inspired metaheuristic
based on the swarming behavior of salps in oceans.

Salps form chains to move effectively through water. The leader at the front
navigates, while followers chain together behind. This behavior is modeled
mathematically for optimization.

Reference:
    Mirjalili, S., Gandomi, A. H., Mirjalili, S. Z., Saremi, S., Faris, H., &
    Mirjalili, S. M. (2017). Salp Swarm Algorithm: A bio-inspired optimizer for
    engineering design problems. Advances in Engineering Software, 114, 163-191.
    DOI: 10.1016/j.advengsoft.2017.07.002

Example:
    >>> from opt.benchmark.functions import shifted_ackley
    >>> optimizer = SalpSwarmOptimizer(
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
    population_size (int): Number of salps in the swarm.
    max_iter (int): Maximum number of iterations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

_LEADER_DIRECTION_THRESHOLD = 0.5


class SalpSwarmOptimizer(AbstractOptimizer):
    """Salp Swarm Algorithm.

    This optimizer mimics the chaining behavior of salps:
    - Salps are divided into leader and followers
    - Leader salp updates position based on food source (best solution)
    - Follower salps follow their predecessor in the chain
    - Coefficient c1 decreases over iterations for convergence

    Attributes:
        seed (int): Random seed for reproducibility.
        lower_bound (float): Lower bound of the search space.
        upper_bound (float): Upper bound of the search space.
        population_size (int): Number of salps.
        dim (int): Problem dimensionality.
        max_iter (int): Maximum iterations.
        func (Callable): Objective function to minimize.


    Example:
        >>> from opt.swarm_intelligence.salp_swarm_algorithm import SalpSwarmOptimizer
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = SalpSwarmOptimizer(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5, max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = SalpSwarmOptimizer(
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
    ) -> None:
        """Initialize the Salp Swarm Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Problem dimensionality.
            max_iter: Maximum iterations.
            seed: Random seed.
            population_size: Number of salps.
        """
        super().__init__(
            func, lower_bound, upper_bound, dim, max_iter, seed, population_size
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Salp Swarm Algorithm.

        Returns:
            Tuple containing:
                - best_solution: The best solution found (numpy array).
                - best_fitness: The fitness value of the best solution.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize salp population
        salps = rng.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

        # Evaluate initial fitness
        fitness = np.array([self.func(salp) for salp in salps])

        # Find food source (best solution)
        best_idx = np.argmin(fitness)
        food_source = salps[best_idx].copy()
        food_fitness = fitness[best_idx]

        # Main optimization loop
        for iteration in range(self.max_iter):
            # Update c1 coefficient (decreases from 2 to 0)
            c1 = 2 * np.exp(-((4 * iteration / self.max_iter) ** 2))

            for i in range(self.population_size):
                if i == 0:
                    # Leader salp position update
                    c2 = rng.random(self.dim)
                    c3 = rng.random(self.dim)

                    # Update leader position based on food source
                    salps[i] = np.where(
                        c3 >= _LEADER_DIRECTION_THRESHOLD,
                        food_source
                        + c1
                        * (
                            (self.upper_bound - self.lower_bound) * c2
                            + self.lower_bound
                        ),
                        food_source
                        - c1
                        * (
                            (self.upper_bound - self.lower_bound) * c2
                            + self.lower_bound
                        ),
                    )
                else:
                    # Follower salp position update (Newton's law of motion)
                    salps[i] = 0.5 * (salps[i] + salps[i - 1])

                # Ensure bounds
                salps[i] = np.clip(salps[i], self.lower_bound, self.upper_bound)

            # Update fitness
            fitness = np.array([self.func(salp) for salp in salps])

            # Update food source
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < food_fitness:
                food_source = salps[best_idx].copy()
                food_fitness = fitness[best_idx]

        return food_source, food_fitness


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(SalpSwarmOptimizer)
