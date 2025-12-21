"""BFGS Optimizer.

This module implements the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm.
BFGS is a quasi-Newton method that approximates Newton's method by using an approximation
to the inverse Hessian matrix. It's particularly effective for smooth optimization problems
and typically converges faster than first-order methods.

BFGS builds up an approximation to the inverse Hessian matrix using gradient information
from previous iterations. This makes it more efficient than computing the actual Hessian
while still providing second-order convergence properties.

This implementation uses scipy's BFGS optimizer with multiple random restarts to improve
global optimization performance.

Example:
    optimizer = BFGS(func=objective_function, lower_bound=-5, upper_bound=5, dim=2)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimensionality of the search space.

Methods:
    search(): Perform the BFGS optimization.
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


class BFGS(AbstractOptimizer):
    """BFGS optimizer implementation using scipy.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        num_restarts (int, optional): Number of random restarts. Defaults to 10.
        seed (int | None, optional): The seed value for random number generation. Defaults to None.


    Example:
        >>> from opt.classical.bfgs import BFGS
        >>> from opt.benchmark.functions import sphere
        >>> optimizer = BFGS(
        ...     func=sphere, dim=2, lower_bound=-5, upper_bound=5,
        ...     max_iter=10, seed=42
        ... )
        >>> solution, fitness = optimizer.search()
        >>> float(fitness) < 100.0  # Should find a reasonable solution
        True

    Example with shifted_ackley:
        >>> from opt.benchmark.functions import shifted_ackley
        >>> optimizer = BFGS(
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
        num_restarts: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialize the BFGS optimizer."""
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
        """Perform the BFGS optimization search with multiple random restarts.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        best_solution = None
        best_fitness = np.inf

        rng = np.random.default_rng(self.seed)

        def bounded_func(x: np.ndarray) -> float:
            """Wrapper function that applies bounds by returning a large penalty if out of bounds."""
            if np.any(x < self.lower_bound) or np.any(x > self.upper_bound):
                return 1e10  # Large penalty for out-of-bounds
            return self.func(x)

        # Perform multiple restarts to improve global optimization
        for _ in range(self.num_restarts):
            # Random starting point
            x0 = rng.uniform(self.lower_bound, self.upper_bound, self.dim)

            try:
                # Use scipy's BFGS optimizer
                result = minimize(
                    fun=bounded_func,
                    x0=x0,
                    method="BFGS",
                    options={"maxiter": self.max_iter // self.num_restarts},
                )

                if result.success and result.fun < best_fitness:
                    # Ensure the solution is within bounds
                    solution = np.clip(result.x, self.lower_bound, self.upper_bound)
                    fitness = self.func(solution)

                    if fitness < best_fitness:
                        best_solution = solution
                        best_fitness = fitness

            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # If optimization fails for this restart, continue with next restart
                continue

        # If no successful optimization, return a random solution
        if best_solution is None:
            best_solution = rng.uniform(self.lower_bound, self.upper_bound, self.dim)
            best_fitness = self.func(best_solution)

        return best_solution, best_fitness


if __name__ == "__main__":
    optimizer = BFGS(func=shifted_ackley, lower_bound=-2.768, upper_bound=+2.768, dim=2)
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
