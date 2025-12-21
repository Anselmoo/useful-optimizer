"""Barrier Method (Interior Point) Optimizer.

This module implements the Barrier Method for constrained optimization,
also known as the Interior Point Method.

The algorithm uses logarithmic barrier functions to keep solutions strictly
inside the feasible region while optimizing the objective.

Reference:
    Boyd, S., & Vandenberghe, L. (2004).
    Convex Optimization.
    Cambridge University Press. Chapter 11: Interior-Point Methods.

Example:
    >>> from opt.benchmark.functions import sphere
    >>> # Minimize sphere with constraint x[0] <= 2
    >>> def constraint(x): return x[0] - 2  # g(x) <= 0 form
    >>> optimizer = BarrierMethodOptimizer(
    ...     func=sphere,
    ...     lower_bound=-5,
    ...     upper_bound=5,
    ...     dim=2,
    ...     constraints=[constraint],
    ...     max_iter=100,
    ... )
    >>> best_solution, best_fitness = optimizer.search()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize

from opt.abstract_optimizer import AbstractOptimizer


if TYPE_CHECKING:
    from collections.abc import Callable


class BarrierMethodOptimizer(AbstractOptimizer):
    """Barrier Method (Interior Point) for constrained optimization.

    This algorithm:
    1. Uses logarithmic barrier for inequality constraints
    2. Progressively reduces barrier coefficient (mu)
    3. Uses gradient-based optimization on barrier objective

    Attributes:
        func: Objective function to minimize.
        lower_bound: Lower bound of search space.
        upper_bound: Upper bound of search space.
        dim: Dimensionality of the problem.
        constraints: List of constraint functions (g(x) <= 0 form).
        max_iter: Maximum outer iterations.
        initial_mu: Starting barrier coefficient.
        mu_reduction: Barrier coefficient reduction factor.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        lower_bound: float,
        upper_bound: float,
        dim: int,
        constraints: list[Callable[[np.ndarray], float]] | None = None,
        max_iter: int = 100,
        initial_mu: float = 10.0,
        mu_reduction: float = 0.5,
    ) -> None:
        """Initialize Barrier Method Optimizer.

        Args:
            func: Objective function to minimize.
            lower_bound: Lower bound of search space.
            upper_bound: Upper bound of search space.
            dim: Dimensionality of the problem.
            constraints: Inequality constraints g(x) <= 0. Defaults to None.
            max_iter: Outer iterations. Defaults to 100.
            initial_mu: Starting barrier coefficient. Defaults to 10.0.
            mu_reduction: Barrier reduction rate. Defaults to 0.5.
        """
        super().__init__(func, lower_bound, upper_bound, dim, max_iter)
        self.constraints = constraints or []
        self.initial_mu = initial_mu
        self.mu_reduction = mu_reduction

    def _barrier_objective(
        self, x: np.ndarray, mu: float
    ) -> float:
        """Compute barrier objective function.

        Args:
            x: Point to evaluate.
            mu: Current barrier coefficient.

        Returns:
            Barrier objective value.
        """
        obj = self.func(x)

        # Logarithmic barrier for inequality constraints
        for g in self.constraints:
            constraint_value = g(x)
            if constraint_value >= 0:
                # Outside feasible region - return large value
                return 1e10
            obj -= mu * np.log(-constraint_value)

        return obj

    def _find_feasible_start(self) -> np.ndarray | None:
        """Find a strictly feasible starting point.

        Returns:
            Feasible point or None if not found.
        """
        # Try random points
        for _ in range(1000):
            x = np.random.uniform(
                self.lower_bound, self.upper_bound, self.dim
            )
            if self._is_strictly_feasible(x):
                return x

        # Try center of bounds
        x = np.full(self.dim, (self.lower_bound + self.upper_bound) / 2)
        if self._is_strictly_feasible(x):
            return x

        return None

    def _is_strictly_feasible(self, x: np.ndarray) -> bool:
        """Check if point is strictly feasible (all g(x) < 0).

        Args:
            x: Point to check.

        Returns:
            True if strictly feasible.
        """
        for g in self.constraints:
            if g(x) >= 0:
                return False
        return True

    def search(self) -> tuple[np.ndarray, float]:
        """Execute the Barrier Method optimization.

        Returns:
            Tuple of (best_solution, best_fitness).
        """
        # Find strictly feasible starting point
        if len(self.constraints) > 0:
            current = self._find_feasible_start()
            if current is None:
                # Fall back to unconstrained optimization
                current = np.random.uniform(
                    self.lower_bound, self.upper_bound, self.dim
                )
        else:
            current = np.random.uniform(
                self.lower_bound, self.upper_bound, self.dim
            )

        bounds = [(self.lower_bound, self.upper_bound)] * self.dim
        mu = self.initial_mu

        best_solution = current.copy()
        best_fitness = self.func(current)

        for _ in range(self.max_iter):
            # Minimize barrier objective
            try:
                result = minimize(
                    lambda x: self._barrier_objective(x, mu),
                    current,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 100},
                )
                if self._is_strictly_feasible(result.x):
                    current = result.x
            except (ValueError, RuntimeWarning):
                # Optimization failed, continue with current point
                pass

            # Update best if feasible
            if self._is_strictly_feasible(current):
                fitness = self.func(current)
                if fitness < best_fitness:
                    best_solution = current.copy()
                    best_fitness = fitness

            # Reduce barrier coefficient
            mu *= self.mu_reduction

            # Termination check
            if mu < 1e-10:
                break

        return best_solution, best_fitness


if __name__ == "__main__":
    from opt.benchmark.functions import sphere

    # Constraint: x[0] <= 1 (i.e., x[0] - 1 <= 0)
    def constraint(x: np.ndarray) -> float:
        return x[0] - 1

    optimizer = BarrierMethodOptimizer(
        func=sphere,
        lower_bound=-5,
        upper_bound=5,
        dim=2,
        constraints=[constraint],
        max_iter=100,
    )
    best_solution, best_fitness = optimizer.search()
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness found: {best_fitness}")
    print(f"Constraint x[0] <= 1: {best_solution[0] <= 1}")
