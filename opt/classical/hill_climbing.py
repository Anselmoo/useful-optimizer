"""Hill Climbing optimizer.

This module implements the Hill Climbing optimizer, which performs a hill climbing
search to find the optimal solution for a given function within the specified bounds.

The HillClimbing class is the main class that implements the optimizer. It takes the
objective function, lower and upper bounds of the search space, dimensionality of the
search space, and other optional parameters as input. The search method performs the
hill climbing search and returns the optimal solution and its corresponding score.

Example usage:
    optimizer = HillClimbing(
        func=shifted_ackley,
        dim=2,
        lower_bound=-32.768,
        upper_bound=+32.768,
        max_iter=5000,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")

Attributes:
    step_sizes (ndarray): The step sizes for each dimension.
    acceleration (float): The acceleration factor.
    epsilon (float): The convergence threshold.
    candidates (ndarray): The candidate locations for each dimension.
    num_candidates (int): The number of candidate locations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class HillClimbing(AbstractOptimizer):
    """Hill Climbing optimizer.

    This optimizer performs a hill climbing search to find the optimal solution
    for a given function within the specified bounds.

    Args:
        func (Callable[[ndarray], float]): The objective function to be minimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        initial_step_sizes (float, optional): The initial step sizes for each dimension. Defaults to 1.0.
        acceleration (float, optional): The acceleration factor. Defaults to 1.2.
        epsilon (float, optional): The convergence threshold. Defaults to 1e-6.
        seed (int | None, optional): The random seed for reproducibility. Defaults to None.



    Example:
        >>> from opt.classical.hill_climbing import HillClimbing
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = HillClimbing(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = HillClimbing(
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
        max_iter: int = 1000,
        initial_step_sizes: float = 1.0,
        acceleration: float = 1.2,
        epsilon: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        """Initialize the HillClimbing class."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.step_sizes = np.full(dim, initial_step_sizes)
        self.acceleration = acceleration
        self.epsilon = epsilon
        self.candidates = np.array(
            [-acceleration, -1 / acceleration, 1 / acceleration, acceleration]
        )
        self.num_candidates = len(self.candidates)

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the hill climbing search.

        Returns:
            tuple[np.ndarray, float]: The optimal solution and its corresponding score.

        """
        # Initialize solution
        solution = np.random.default_rng(self.seed).uniform(
            self.lower_bound, self.upper_bound, self.dim
        )
        best_score = self.func(solution)

        for _ in range(self.max_iter):
            before_score = best_score
            for i in range(self.dim):
                before_point = solution[i]
                best_step = 0
                for j in range(
                    self.num_candidates
                ):  # try each of the candidate locations
                    step = self.step_sizes[i] * self.candidates[j]
                    solution[i] = before_point + step
                    score = self.func(solution)
                    if score < best_score:  # assuming we are minimizing the function
                        best_score = score
                        best_step = step
                if best_step == 0:
                    solution[i] = before_point
                    self.step_sizes[i] /= self.acceleration
                else:
                    solution[i] = before_point + best_step
                    self.step_sizes[i] = best_step  # acceleration
            if abs(best_score - before_score) < self.epsilon:
                break

        return solution, best_score


if __name__ == "__main__":
    optimizer = HillClimbing(
        func=shifted_ackley, dim=2, lower_bound=-32.768, upper_bound=+32.768
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
