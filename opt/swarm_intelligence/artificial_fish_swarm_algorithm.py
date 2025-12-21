"""Artificial Fish Swarm Algorithm (AFSA).

This module implements the Artificial Fish Swarm Algorithm (AFSA). AFSA is a population
based optimization technique inspired by the social behavior of fishes. In their social
behavior, fish try to keep a balance between food consistency and crowding effect. This
behavior is modeled into a mathematical optimization technique in AFSA.

In AFSA, each fish represents a potential solution and the food consistency represents
the objective function to be optimized. Each fish tries to move towards better regions
of the search space based on its own experience and the experience of its neighbors.

AFSA has been used for various kinds of optimization problems including function
optimization, neural network training, fuzzy system control, and other areas of
engineering.

Example:
    from opt.artificial_fish_swarm_algorithm import ArtificialFishSwarm
    optimizer = ArtificialFishSwarm(func=objective_function, lower_bound=-10,
    upper_bound=10, dim=2, n_fish=50, max_iter=1000)
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimension of the search space.
    n_fish (int): The number of fish (candidate solutions).
    max_iter (int): The maximum number of iterations.

Methods:
    search(): Perform the AFSA optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ArtificialFishSwarm(AbstractOptimizer):
    """Implementation of the Artificial Fish Swarm algorithm for optimization.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        fish_swarm (int): The number of fish in the swarm (default: 50).
        max_iter (int): The maximum number of iterations (default: 1000).
        visual (int): The visual range of each fish (default: 1).
        step (float): The step size for fish movement (default: 0.1).
        try_number (int): The number of attempts for prey behavior (default: 3).
        epsilon (float): A small value added to avoid division by zero (default: 1e-9).
        seed (Optional[int]): The seed for the random number generator (default: None).

    Attributes:
        visual (int): The visual range of each fish.
        step (float): The step size for fish movement.
        try_number (int): The number of attempts for prey behavior.
        epsilon (float): A small value added to avoid division by zero.
        fishes (np.ndarray): The current positions of the fish in the swarm.

    Methods:
        search() -> Tuple[np.ndarray, float]:
            Run the optimization process and return the best solution found.

        behavior(i: int) -> np.ndarray:
            Perform the behavior of the fish at index i.

    Inherits from:
        AbstractOptimizer: An abstract base class for optimization algorithms.


    Example:
        >>> from opt.swarm_intelligence.artificial_fish_swarm_algorithm import ArtificialFishSwarm
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = ArtificialFishSwarm(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = ArtificialFishSwarm(
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
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        fish_swarm: int = 50,
        max_iter: int = 1000,
        visual: int = 1,
        step: float = 0.1,
        try_number: int = 3,
        epsilon: float = 1e-9,
        seed: int | None = None,
    ) -> None:
        """Initialize the Artificial Fish Swarm algorithm."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
            population_size=fish_swarm,
        )
        self.visual = visual
        self.step = step
        self.try_number = try_number
        self.epsilon = epsilon
        self.fishes = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )

    def search(self) -> tuple[np.ndarray, float]:
        """Run the optimization process and return the best solution found.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the best solution found and its corresponding fitness value.
        """
        best_fitness = np.inf
        best_solution = None

        for _ in range(self.max_iter):
            self.seed += 1
            for i in range(self.population_size):
                solution = self.fishes[i]
                fitness = self.func(solution)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = solution

                self.fishes[i] = self.behavior(i)

        return best_solution, best_fitness

    def behavior(self, i: int) -> np.ndarray:
        """Perform the behavior of the fish at index i.

        Args:
            i (int): The index of the fish.

        Returns:
            np.ndarray: The updated position of the fish.
        """
        # Prey behavior
        for i in range(self.try_number):
            new_solution = self.fishes[i] + self.step * np.random.default_rng(
                self.seed + i
            ).uniform(-1, 1, self.dim)
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            if self.func(new_solution) < self.func(self.fishes[i]):
                return new_solution

        # Swarm behavior
        neighbors = self.fishes[
            np.linalg.norm(self.fishes - self.fishes[i], axis=1) < self.visual
        ]
        center = np.mean(neighbors, axis=0)
        new_solution = self.fishes[i] + self.step * (center - self.fishes[i]) / (
            np.linalg.norm(center - self.fishes[i]) + self.epsilon
        )
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if self.func(new_solution) < self.func(self.fishes[i]):
            return new_solution

        # Follow behavior
        best_neighbor = neighbors[np.argmin([self.func(x) for x in neighbors])]
        new_solution = self.fishes[i] + self.step * (best_neighbor - self.fishes[i]) / (
            np.linalg.norm(best_neighbor - self.fishes[i]) + self.epsilon
        )
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if self.func(new_solution) < self.func(self.fishes[i]):
            return new_solution

        return self.fishes[i]


if __name__ == "__main__":
    from opt.demo import run_demo

    run_demo(ArtificialFishSwarm)
