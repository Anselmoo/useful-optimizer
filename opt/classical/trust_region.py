"""Trust Region Optimizer.

This module implements Trust Region optimization algorithms. Trust region methods
are a class of optimization algorithms that work by defining a region around the
current point where a model (usually quadratic) of the objective function is trusted
to be accurate. The algorithm finds the step that minimizes the model within this
trust region.

Trust region methods have several advantages:
- They are globally convergent under reasonable assumptions
- They automatically adapt the step size based on the quality of the model
- They handle ill-conditioned problems better than line search methods
- They are robust to numerical difficulties

This implementation provides access to scipy's trust region methods including:
- trust-constr: Trust region method with constraints
- trust-exact: Trust region method with exact Hessian
- trust-krylov: Trust region method using Krylov subspace

Example:
    optimizer = TrustRegion(func=objective_function, lower_bound=-5, upper_bound=5, dim=2)
    best_solution, best_fitness = optimizer.search()

Attributes:
    func (Callable): The objective function to optimize.
    lower_bound (float): The lower bound of the search space.
    upper_bound (float): The upper bound of the search space.
    dim (int): The dimensionality of the search space.

Methods:
    search(): Perform the Trust Region optimization.
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


class TrustRegion(AbstractOptimizer):
    """Trust Region optimizer implementation using scipy.

    Args:
        func (Callable[[ndarray], float]): The objective function to be optimized.
        lower_bound (float): The lower bound of the search space.
        upper_bound (float): The upper bound of the search space.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        num_restarts (int, optional): Number of random restarts. Defaults to 10.
        method (str, optional): Trust region method to use. Defaults to 'trust-constr'.
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
        method: str = "trust-constr",
        seed: int | None = None,
    ) -> None:
        """Initialize the Trust Region optimizer."""
        super().__init__(
            func=func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dim=dim,
            max_iter=max_iter,
            seed=seed,
        )
        self.num_restarts = num_restarts
        self.method = method

        # Validate method choice
        valid_methods = ["trust-constr", "trust-exact", "trust-krylov", "trust-ncg"]
        if self.method not in valid_methods:
            msg = f"Method must be one of {valid_methods}, got {self.method}"
            raise ValueError(msg)

    def search(self) -> tuple[np.ndarray, float]:
        """Perform the Trust Region optimization search with multiple random restarts.

        Returns:
            tuple[np.ndarray, float]: A tuple containing the best solution found and its fitness value.
        """
        best_solution = None
        best_fitness = np.inf

        rng = np.random.default_rng(self.seed)

        # Perform multiple restarts to improve global optimization
        for _ in range(self.num_restarts):
            # Random starting point
            x0 = rng.uniform(self.lower_bound, self.upper_bound, self.dim)

            try:
                # Set up bounds for trust-constr method
                if self.method == "trust-constr":
                    bounds = [(self.lower_bound, self.upper_bound)] * self.dim
                    result = minimize(
                        fun=self.func,
                        x0=x0,
                        method=self.method,
                        bounds=bounds,
                        options={"maxiter": self.max_iter // self.num_restarts},
                    )
                else:
                    # For other trust region methods, use penalty for bounds
                    def bounded_func(x: np.ndarray) -> float:
                        if np.any(x < self.lower_bound) or np.any(x > self.upper_bound):
                            return 1e10  # Large penalty for out-of-bounds
                        return self.func(x)

                    result = minimize(
                        fun=bounded_func,
                        x0=x0,
                        method=self.method,
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
    optimizer = TrustRegion(
        func=shifted_ackley, lower_bound=-2.768, upper_bound=+2.768, dim=2
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
