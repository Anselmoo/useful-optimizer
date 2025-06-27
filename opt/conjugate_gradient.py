"""Conjugate Gradient Optimizer.

This module implements the Conjugate Gradient optimization algorithm. The Conjugate
Gradient method is an algorithm for the numerical solution of systems of linear
equations whose matrix is positive-definite. For general optimization, it's used
as an iterative method for solving unconstrained optimization problems.

The method works by:
1. Computing the gradient at the current point
2. Determining a conjugate direction (orthogonal in a specific sense)
3. Performing a line search along this direction
4. Updating the position and computing a new conjugate direction

The conjugate gradient method has the property that it converges in at most n steps
for a quadratic function in n dimensions, making it particularly effective for
quadratic and near-quadratic problems.

This implementation uses scipy's CG optimizer with multiple random restarts
to improve global optimization performance.

Example:
    optimizer = ConjugateGradient(func=objective_function, lower_bound=-5, upper_bound=5, dim=2)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimensionality of the search space.

Methods:
    search(): Perform the Conjugate Gradient optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize

from opt.abstract_optimizer import AbstractOptimizer
from opt.benchmark.functions import shifted_ackley


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import ndarray


class ConjugateGradient(AbstractOptimizer):
    """Conjugate Gradient optimizer implementation using scipy.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        num_restarts (int, optional): Number of random restarts. Defaults to 10.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.
    """

    def __init__(
        self,
        func: Callable[[ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        max_iter: int = 1000,
        num_restarts: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize the Conjugate Gradient optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.num_restarts = num_restarts

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the Conjugate Gradient optimization search with multiple random restarts.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        best_solution = None
        best_fitness = np.inf

        rng = np.random.default_rng(self.seed)

        # Perform multiple restarts to improve global optimization
        for i in range(self.num_restarts):
            # Random starting point
            x0 = rng.uniform(self.lower_bound, self.upper_bound, self.dim)

            try:
                # Use scipy's Conjugate Gradient optimizer
                result = minimize(
                    fun=self.func,
                    x0=x0,
                    method="CG",
                    bounds=[(self.lower_bound, self.upper_bound)] * self.dim,
                    options={"maxiter": self.max_iter // self.num_restarts}
                )

                if result.success and result.fun < best_fitness:
                    # Ensure the solution is within bounds
                    solution = np.clip(result.x, self.lower_bound, self.upper_bound)
                    fitness = self.func(solution)

                    if fitness < best_fitness:
                        best_solution = solution
                        best_fitness = fitness

            except Exception:
                # If optimization fails for this restart, continue with next restart
                continue

        # If no successful optimization, return a random solution
        if best_solution is None:
            best_solution = rng.uniform(self.lower_bound, self.upper_bound, self.dim)
            best_fitness = self.func(best_solution)

        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = ConjugateGradient(
        func=shifted_ackley, lower_bound=-2.768, upper_bound=+2.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
